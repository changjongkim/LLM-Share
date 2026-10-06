// A serving process that continues one prefix with several sequences at
// once, as a batching server does.
//
//   kv_batch alone  MODEL CTX PREFIX_FILE SEQUENCES FIRST_AGENT N_GEN
//   kv_batch parent MODEL CTX PREFIX_FILE STATE SEQUENCES FIRST_AGENT N_GEN
//   kv_batch child  MODEL CTX STATE SEQUENCES FIRST_AGENT N_GEN
//
// `alone` computes the prefix itself. `parent` computes it, saves the state
// of the context to STATE, prints a PUBLISHED line, and continues. `child`
// loads STATE. Every role then shares the prefix between its sequences
// inside the process (the cells of the prefix get every sequence id; no row
// is copied), appends the task of agent FIRST_AGENT + i to sequence i,
// decodes the tasks as one batch, and generates N_GEN tokens per sequence,
// one batch of SEQUENCES tokens per step, taking the most probable token.
//
// Where the key-value cache lives is decided by the engine from
// LLAMA_KV_HOST, LLAMA_KV_PREFIX and LLAMA_KV_GROW, as for kv_fork.
#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read " + path);
  }
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text, bool special) {
  const int needed = -llama_tokenize(vocab, text.data(),
                                     static_cast<int32_t>(text.size()),
                                     nullptr, 0, special, false);
  std::vector<llama_token> tokens(static_cast<size_t>(std::max(needed, 0)));
  const int written =
      llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                     tokens.data(), static_cast<int32_t>(tokens.size()),
                     special, false);
  if (written != needed) {
    throw std::runtime_error("tokenization failed");
  }
  return tokens;
}

// The task of an agent; the same text as in the campaigns with one agent per
// process.
std::string task_of(int agent) {
  return "\n\nTask for agent " + std::to_string(agent) +
         ": summarize the text above in " + std::to_string(agent + 2) +
         " sentences.";
}

// One token of one sequence in a batch.
void add(llama_batch& batch, llama_token token, llama_pos position,
         llama_seq_id sequence, bool output) {
  const int32_t at = batch.n_tokens++;
  batch.token[at] = token;
  batch.pos[at] = position;
  batch.n_seq_id[at] = 1;
  batch.seq_id[at][0] = sequence;
  batch.logits[at] = output ? 1 : 0;
}

llama_token most_probable(llama_context* context, int32_t at, int vocabulary) {
  const float* logits = llama_get_logits_ith(context, at);
  return static_cast<llama_token>(
      std::max_element(logits, logits + vocabulary) - logits);
}

std::string one_line(const std::string& text) {
  std::string line;
  for (const char c : text) {
    if (c == '\\') {
      line += "\\\\";
    } else if (c == '\n') {
      line += "\\n";
    } else if (c == '\r') {
      line += "\\r";
    } else {
      line += c;
    }
  }
  return line;
}

int run(int argc, char** argv) {
  const auto process_start = Clock::now();
  const std::string role = argc > 1 ? argv[1] : "";
  const bool alone = role == "alone";
  const bool parent = role == "parent";
  const bool child = role == "child";
  const int expected = alone ? 8 : parent ? 9 : child ? 8 : 0;
  if (expected == 0 || argc != expected) {
    std::fprintf(
        stderr,
        "usage: kv_batch alone  MODEL CTX PREFIX_FILE SEQUENCES FIRST_AGENT "
        "N_GEN\n"
        "       kv_batch parent MODEL CTX PREFIX_FILE STATE SEQUENCES "
        "FIRST_AGENT N_GEN\n"
        "       kv_batch child  MODEL CTX STATE SEQUENCES FIRST_AGENT N_GEN\n");
    return 2;
  }
  const std::string model_path = argv[2];
  const uint32_t context_size =
      static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  const std::string prefix_path = child ? "" : argv[4];
  const std::string state_path = child ? argv[4] : parent ? argv[5] : "";
  const int sequences = std::atoi(argv[expected - 3]);
  const int first_agent = std::atoi(argv[expected - 2]);
  const int n_gen = std::atoi(argv[expected - 1]);
  const size_t batch_size = 2048;
  if (sequences < 1 || n_gen < 1) {
    throw std::runtime_error("SEQUENCES and N_GEN must be positive");
  }

  llama_backend_init();
  auto mark = Clock::now();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 99;
  llama_model* model =
      llama_model_load_from_file(model_path.c_str(), model_params);
  if (model == nullptr) {
    throw std::runtime_error("cannot load " + model_path);
  }
  const double model_ms = ms_since(mark);
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int vocabulary = llama_vocab_n_tokens(vocab);

  mark = Clock::now();
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = context_size;
  context_params.n_batch = static_cast<uint32_t>(batch_size);
  context_params.n_seq_max = static_cast<uint32_t>(sequences);
  // One cache for all sequences: the cells of the prefix are shared by them.
  context_params.kv_unified = true;
  context_params.no_perf = true;
  llama_context* context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    throw std::runtime_error("cannot create the context");
  }
  const double context_ms = ms_since(mark);

  std::vector<llama_token> prefix;
  double prefix_ms = 0.0;
  double state_ms = 0.0;
  if (child) {
    mark = Clock::now();
    prefix.resize(context_size);
    size_t count = 0;
    if (!llama_state_load_file(context, state_path.c_str(), prefix.data(),
                               prefix.size(), &count)) {
      throw std::runtime_error("cannot load the state " + state_path);
    }
    prefix.resize(count);
    state_ms = ms_since(mark);
  } else {
    prefix = tokenize(vocab, read_file(prefix_path), true);
    mark = Clock::now();
    for (size_t done = 0; done < prefix.size(); done += batch_size) {
      const size_t piece = std::min(batch_size, prefix.size() - done);
      if (llama_decode(context,
                       llama_batch_get_one(prefix.data() + done,
                                           static_cast<int32_t>(piece))) != 0) {
        throw std::runtime_error("decode failed");
      }
    }
    llama_synchronize(context);
    prefix_ms = ms_since(mark);
  }
  const llama_pos prefix_tokens = static_cast<llama_pos>(prefix.size());

  if (parent) {
    mark = Clock::now();
    if (!llama_state_save_file(context, state_path.c_str(), prefix.data(),
                               prefix.size())) {
      throw std::runtime_error("cannot save the state " + state_path);
    }
    state_ms = ms_since(mark);
    struct stat status {};
    const long long state_bytes =
        stat(state_path.c_str(), &status) == 0
            ? static_cast<long long>(status.st_size)
            : -1;
    std::printf("PUBLISHED prefix_tokens=%d prefix_ms=%.1f publish_ms=%.2f "
                "state_bytes=%lld since_start_ms=%.1f\n",
                prefix_tokens, prefix_ms, state_ms, state_bytes,
                ms_since(process_start));
    std::fflush(stdout);
  }

  // Every sequence continues the prefix: its cells get all sequence ids.
  mark = Clock::now();
  llama_memory_t memory = llama_get_memory(context);
  for (int sequence = 1; sequence < sequences; ++sequence) {
    llama_memory_seq_cp(memory, 0, sequence, -1, -1);
  }
  std::vector<llama_pos> position(static_cast<size_t>(sequences),
                                  prefix_tokens);
  std::vector<int32_t> output(static_cast<size_t>(sequences), -1);
  llama_batch batch =
      llama_batch_init(static_cast<int32_t>(batch_size), 0, 1);
  for (int sequence = 0; sequence < sequences; ++sequence) {
    const std::vector<llama_token> task =
        tokenize(vocab, task_of(first_agent + sequence), false);
    if (batch.n_tokens + static_cast<int32_t>(task.size()) >
        static_cast<int32_t>(batch_size)) {
      throw std::runtime_error("the tasks do not fit into one batch");
    }
    for (size_t i = 0; i < task.size(); ++i) {
      const bool last = i + 1 == task.size();
      if (last) {
        output[static_cast<size_t>(sequence)] = batch.n_tokens;
      }
      add(batch, task[i], position[static_cast<size_t>(sequence)]++, sequence,
          last);
    }
  }
  if (llama_decode(context, batch) != 0) {
    throw std::runtime_error("decode failed");
  }
  llama_synchronize(context);
  const double suffix_ms = ms_since(mark);

  mark = Clock::now();
  std::vector<std::string> text(static_cast<size_t>(sequences));
  double first_token_ms = 0.0;
  for (int step = 0; step < n_gen; ++step) {
    std::vector<llama_token> next(static_cast<size_t>(sequences));
    for (int sequence = 0; sequence < sequences; ++sequence) {
      const size_t at = static_cast<size_t>(sequence);
      next[at] = most_probable(context, output[at], vocabulary);
      char piece[256];
      const int length =
          llama_token_to_piece(vocab, next[at], piece, sizeof(piece), 0, true);
      if (length > 0) {
        text[at].append(piece, static_cast<size_t>(length));
      }
    }
    if (step == 0) {
      first_token_ms = ms_since(process_start);
    }
    batch.n_tokens = 0;
    for (int sequence = 0; sequence < sequences; ++sequence) {
      const size_t at = static_cast<size_t>(sequence);
      output[at] = batch.n_tokens;
      add(batch, next[at], position[at]++, sequence, true);
    }
    if (llama_decode(context, batch) != 0) {
      throw std::runtime_error("decode failed");
    }
  }
  llama_synchronize(context);
  const double generation_ms = ms_since(mark);
  const int generated = sequences * n_gen;

  std::printf(
      "RESULT role=%s sequences=%d first_agent=%d prefix_tokens=%d "
      "model_ms=%.1f context_ms=%.1f state_ms=%.2f prefix_ms=%.1f "
      "suffix_ms=%.1f first_token_ms=%.1f generation_ms=%.1f generated=%d "
      "generation_tps=%.3f\n",
      role.c_str(), sequences, first_agent, prefix_tokens, model_ms,
      context_ms, state_ms, prefix_ms, suffix_ms, first_token_ms,
      generation_ms, generated,
      generation_ms > 0.0 ? generated * 1000.0 / generation_ms : 0.0);
  for (int sequence = 0; sequence < sequences; ++sequence) {
    std::printf("TEXT %d %s\n", first_agent + sequence,
                one_line(text[static_cast<size_t>(sequence)]).c_str());
  }
  std::fflush(stdout);

  llama_batch_free(batch);
  llama_free(context);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "kv_batch: %s\n", error.what());
    return 1;
  }
}
