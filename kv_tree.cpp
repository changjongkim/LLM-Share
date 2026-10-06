// One process of a tree of agents. It runs a list of steps on one context:
//
//   kv_tree MODEL CTX N_GEN STEP...
//
//   text:FILE    decode the tokens of FILE
//   say:STRING   decode the tokens of STRING
//   load:STATE   restore the context from STATE and print an ATTACHED line
//   save:STATE   save the context to STATE and print a PUBLISHED line
//   wait:FILE    wait until FILE exists
//   gen          generate N_GEN tokens greedily
//
// A root is "text:PREFIX save:S0 wait:GO say:TASK gen", an inner agent is
// "load:S0 say:CONTEXT save:S1 wait:GO say:TASK gen", and a leaf is
// "load:S1 say:TASK gen". Where the rows of the cache live follows from the
// environment of the engine (LLAMA_KV_HOST, LLAMA_KV_CHAIN, LLAMA_KV_GROW):
// with none of them a state file carries the rows and every process holds a
// copy, with them a state file names rows that the next process maps.
#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: kv_tree MODEL CTX N_GEN STEP...\n"
                 "steps: text:FILE say:STRING load:STATE save:STATE "
                 "wait:FILE gen\n");
    return 2;
  }
  const std::string model_path = argv[1];
  const uint32_t context_size =
      static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  const int n_gen = std::atoi(argv[3]);
  const size_t batch = 2048;

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
  context_params.n_batch = static_cast<uint32_t>(batch);
  context_params.no_perf = true;
  llama_context* context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    throw std::runtime_error("cannot create the context");
  }
  const double context_ms = ms_since(mark);

  std::vector<llama_token> tokens;
  double load_ms = 0.0;
  double decode_ms = 0.0;
  double save_ms = 0.0;
  double generation_ms = 0.0;
  double first_token_ms = 0.0;
  size_t loaded_tokens = 0;
  int generated = 0;
  std::string text;

  for (int index = 4; index < argc; ++index) {
    const std::string step = argv[index];
    const size_t colon = step.find(':');
    const std::string kind = step.substr(0, colon);
    const std::string value =
        colon == std::string::npos ? "" : step.substr(colon + 1);
    mark = Clock::now();
    if (kind == "text" || kind == "say") {
      // The first tokens of a context carry the special tokens of the model.
      const std::vector<llama_token> added = tokenize(
          vocab, kind == "text" ? read_file(value) : value, tokens.empty());
      const size_t first = tokens.size();
      tokens.insert(tokens.end(), added.begin(), added.end());
      for (size_t done = 0; done < added.size(); done += batch) {
        const size_t piece = std::min(batch, added.size() - done);
        if (llama_decode(context,
                         llama_batch_get_one(tokens.data() + first + done,
                                             static_cast<int32_t>(piece))) !=
            0) {
          throw std::runtime_error("decode failed");
        }
      }
      llama_synchronize(context);
      decode_ms += ms_since(mark);
    } else if (kind == "load") {
      tokens.resize(context_size);
      size_t count = 0;
      if (!llama_state_load_file(context, value.c_str(), tokens.data(),
                                 tokens.size(), &count)) {
        throw std::runtime_error("cannot load the state " + value);
      }
      tokens.resize(count);
      loaded_tokens = count;
      load_ms += ms_since(mark);
      std::printf("ATTACHED tokens=%zu context_ms=%.1f load_ms=%.2f "
                  "since_start_ms=%.1f\n",
                  count, context_ms, load_ms, ms_since(process_start));
      std::fflush(stdout);
    } else if (kind == "save") {
      if (!llama_state_save_file(context, value.c_str(), tokens.data(),
                                 tokens.size())) {
        throw std::runtime_error("cannot save the state " + value);
      }
      const double this_save_ms = ms_since(mark);
      save_ms += this_save_ms;
      struct stat status {};
      const long long state_bytes =
          stat(value.c_str(), &status) == 0
              ? static_cast<long long>(status.st_size)
              : -1;
      std::printf("PUBLISHED tokens=%zu publish_ms=%.2f state_bytes=%lld\n",
                  tokens.size(), this_save_ms, state_bytes);
      std::fflush(stdout);
    } else if (kind == "wait") {
      for (;;) {
        struct stat go {};
        if (stat(value.c_str(), &go) == 0) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } else if (kind == "gen") {
      for (int made = 0; made < n_gen; ++made) {
        const float* logits = llama_get_logits_ith(context, -1);
        llama_token token = static_cast<llama_token>(
            std::max_element(logits, logits + vocabulary) - logits);
        if (generated == 0) {
          first_token_ms = ms_since(process_start);
        }
        char piece[256];
        const int length =
            llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (length > 0) {
          text.append(piece, static_cast<size_t>(length));
        }
        tokens.push_back(token);
        if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) {
          throw std::runtime_error("decode failed");
        }
        ++generated;
      }
      llama_synchronize(context);
      generation_ms += ms_since(mark);
    } else {
      throw std::runtime_error("unknown step " + step);
    }
  }

  std::printf(
      "RESULT tokens=%zu loaded_tokens=%zu model_ms=%.1f context_ms=%.1f "
      "state_ms=%.2f decode_ms=%.1f save_ms=%.2f first_token_ms=%.1f "
      "generation_ms=%.1f generated=%d generation_tps=%.3f total_ms=%.1f\n",
      tokens.size(), loaded_tokens, model_ms, context_ms, load_ms, decode_ms,
      save_ms, first_token_ms, generation_ms, generated,
      generation_ms > 0.0 ? generated * 1000.0 / generation_ms : 0.0,
      ms_since(process_start));
  std::printf("TEXT %s\n", one_line(text).c_str());
  std::fflush(stdout);

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
    std::fprintf(stderr, "kv_tree: %s\n", error.what());
    return 1;
  }
}
