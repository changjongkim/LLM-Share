// One role of a prefix-sharing experiment on the engine.
//
//   kv_fork alone  MODEL CTX PREFIX_FILE SUFFIX N_GEN
//   kv_fork parent MODEL CTX PREFIX_FILE STATE SUFFIX N_GEN [GO_FILE]
//   kv_fork child  MODEL CTX STATE SUFFIX N_GEN
//
// `alone` computes the prefix itself. `parent` computes the prefix, saves the
// state of the context to STATE, prints a PUBLISHED line, and then, when
// N_GEN is not zero, continues with its own suffix (after GO_FILE exists, if
// one is named). `child` loads STATE and continues with its suffix.
//
// Where the key-value cache lives, and whether STATE holds the rows or only
// names them, is decided by the engine from LLAMA_KV_HOST, LLAMA_KV_PREFIX,
// LLAMA_KV_COW and LLAMA_KV_GROW; this program is the same in every mode.
// Every role decodes the same batches (the prefix in pieces of the batch
// size, the suffix as one batch, then one token at a time) and takes the
// most probable token, so the texts of all modes can be compared.
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

void decode(llama_context* context, std::vector<llama_token>& tokens,
            size_t first, size_t count, size_t batch) {
  for (size_t done = 0; done < count; done += batch) {
    const size_t piece = std::min(batch, count - done);
    if (llama_decode(context,
                     llama_batch_get_one(tokens.data() + first + done,
                                         static_cast<int32_t>(piece))) != 0) {
      throw std::runtime_error("decode failed");
    }
  }
  // The engine returns before the device is done; a phase ends when it is.
  llama_synchronize(context);
}

llama_token most_probable(llama_context* context, int vocabulary) {
  const float* logits = llama_get_logits_ith(context, -1);
  return static_cast<llama_token>(
      std::max_element(logits, logits + vocabulary) - logits);
}

// Text on one line: a backslash escapes itself and the line break.
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
  const int fixed = alone ? 7 : parent ? 8 : child ? 7 : 0;
  if (fixed == 0 || argc < fixed || argc > fixed + (parent ? 1 : 0)) {
    std::fprintf(stderr,
                 "usage: kv_fork alone  MODEL CTX PREFIX_FILE SUFFIX N_GEN\n"
                 "       kv_fork parent MODEL CTX PREFIX_FILE STATE SUFFIX "
                 "N_GEN [GO_FILE]\n"
                 "       kv_fork child  MODEL CTX STATE SUFFIX N_GEN\n");
    return 2;
  }
  const std::string model_path = argv[2];
  const uint32_t context_size =
      static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  const std::string prefix_path = child ? "" : argv[4];
  const std::string state_path = child ? argv[4] : parent ? argv[5] : "";
  const std::string suffix = argv[fixed - 2];
  const int n_gen = std::atoi(argv[fixed - 1]);
  const std::string go_path = argc > fixed ? argv[fixed] : "";
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
  double prefix_ms = 0.0;
  double state_ms = 0.0;
  if (child) {
    mark = Clock::now();
    tokens.resize(context_size);
    size_t count = 0;
    if (!llama_state_load_file(context, state_path.c_str(), tokens.data(),
                               tokens.size(), &count)) {
      throw std::runtime_error("cannot load the state " + state_path);
    }
    tokens.resize(count);
    state_ms = ms_since(mark);
  } else {
    tokens = tokenize(vocab, read_file(prefix_path), true);
    mark = Clock::now();
    decode(context, tokens, 0, tokens.size(), batch);
    prefix_ms = ms_since(mark);
  }
  const size_t prefix_tokens = tokens.size();

  if (parent) {
    mark = Clock::now();
    if (!llama_state_save_file(context, state_path.c_str(), tokens.data(),
                               tokens.size())) {
      throw std::runtime_error("cannot save the state " + state_path);
    }
    state_ms = ms_since(mark);
    struct stat status {};
    const long long state_bytes =
        stat(state_path.c_str(), &status) == 0
            ? static_cast<long long>(status.st_size)
            : -1;
    std::printf(
        "PUBLISHED prefix_tokens=%zu prefix_ms=%.1f publish_ms=%.2f "
        "state_bytes=%lld\n",
        prefix_tokens, prefix_ms, state_ms, state_bytes);
    std::fflush(stdout);
    if (n_gen == 0) {
      llama_free(context);
      llama_model_free(model);
      llama_backend_free();
      return 0;
    }
    while (!go_path.empty()) {
      struct stat go {};
      if (stat(go_path.c_str(), &go) == 0) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  mark = Clock::now();
  const std::vector<llama_token> suffix_tokens =
      tokenize(vocab, suffix, false);
  tokens.insert(tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
  decode(context, tokens, prefix_tokens, suffix_tokens.size(), batch);
  const double suffix_ms = ms_since(mark);

  mark = Clock::now();
  std::string text;
  double first_token_ms = 0.0;
  for (int step = 0; step < n_gen; ++step) {
    llama_token token = most_probable(context, vocabulary);
    if (step == 0) {
      first_token_ms = ms_since(process_start);
    }
    char piece[256];
    const int length =
        llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
    if (length > 0) {
      text.append(piece, static_cast<size_t>(length));
    }
    if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) {
      throw std::runtime_error("decode failed");
    }
  }
  llama_synchronize(context);
  const double generation_ms = ms_since(mark);

  std::printf(
      "RESULT role=%s prefix_tokens=%zu suffix_tokens=%zu model_ms=%.1f "
      "context_ms=%.1f state_ms=%.2f prefix_ms=%.1f suffix_ms=%.1f "
      "first_token_ms=%.1f generation_ms=%.1f generated=%d "
      "generation_tps=%.3f\n",
      role.c_str(), prefix_tokens, suffix_tokens.size(), model_ms, context_ms,
      state_ms, prefix_ms, suffix_ms, first_token_ms, generation_ms, n_gen,
      generation_ms > 0.0 ? n_gen * 1000.0 / generation_ms : 0.0);
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
    std::fprintf(stderr, "kv_fork: %s\n", error.what());
    return 1;
  }
}
