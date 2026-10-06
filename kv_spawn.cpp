// A serving process that computes a prefix, publishes it, and starts its
// agents itself.
//
//   kv_spawn MODEL CTX PREFIX_FILE STATE N_GEN GO_FILE OUT_DIR CHILD_BINARY
//            CHILD_MIG...
//
// The parent computes the prefix, saves the state of its context to STATE,
// prints a PUBLISHED line, and waits for GO_FILE. Then it starts one child
// per CHILD_MIG (CHILD_BINARY child MODEL CTX STATE TASK N_GEN, with
// CUDA_VISIBLE_DEVICES set to that MIG instance, output in
// OUT_DIR/child.INDEX), generates N_GEN tokens of its own continuation while
// the children generate theirs, and waits for them.
//
// The children are started by the parent, and not by a script, because one
// of the ways to hand the cache over needs it: device memory that is shared
// through file descriptors, which a child inherits. What the children
// receive follows from the cache mode of the parent's environment:
//   LLAMA_KV_VMM=1     the descriptors that the engine exported at the save
//                      (LLAMA_KV_VMM_EXPORT becomes LLAMA_KV_VMM_IMPORT)
//   LLAMA_KV_HOST=PATH the number of published rows (LLAMA_KV_PREFIX)
//   neither            nothing; STATE holds the rows and the child copies them
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

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

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
  llama_synchronize(context);
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

// The task of a child; the same text as in the other campaigns.
std::string task_of(int agent) {
  return "\n\nTask for agent " + std::to_string(agent) +
         ": summarize the text above in " + std::to_string(agent + 2) +
         " sentences.";
}

// The environment of a child: this process's, with the listed variables
// replaced or added.
std::vector<std::string> child_environment(
    const std::vector<std::pair<std::string, std::string>>& changes) {
  std::vector<std::string> result;
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string line = *entry;
    const std::string name = line.substr(0, line.find('='));
    const bool replaced =
        std::any_of(changes.begin(), changes.end(),
                    [&name](const auto& change) { return change.first == name; });
    if (!replaced && name != "LLAMA_KV_VMM_EXPORT") {
      result.push_back(line);
    }
  }
  for (const auto& [name, value] : changes) {
    result.push_back(name + "=" + value);
  }
  return result;
}

pid_t start_child(const std::string& binary,
                  const std::vector<std::string>& arguments,
                  const std::vector<std::string>& environment,
                  const std::string& output) {
  std::vector<char*> argv;
  for (const auto& argument : arguments) {
    argv.push_back(const_cast<char*>(argument.c_str()));
  }
  argv.push_back(nullptr);
  std::vector<char*> envp;
  for (const auto& entry : environment) {
    envp.push_back(const_cast<char*>(entry.c_str()));
  }
  envp.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, output.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  const std::string errors = output + ".err";
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, errors.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  pid_t pid = -1;
  const int status = posix_spawn(&pid, binary.c_str(), &actions, nullptr,
                                 argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&actions);
  if (status != 0) {
    throw std::runtime_error("cannot start " + binary);
  }
  return pid;
}

int run(int argc, char** argv) {
  const auto process_start = Clock::now();
  if (argc < 10) {
    std::fprintf(stderr,
                 "usage: kv_spawn MODEL CTX PREFIX_FILE STATE N_GEN GO_FILE "
                 "OUT_DIR CHILD_BINARY CHILD_MIG...\n");
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string context_text = argv[2];
  const uint32_t context_size =
      static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  const std::string prefix_path = argv[3];
  const std::string state_path = argv[4];
  const std::string generate_text = argv[5];
  const int n_gen = std::atoi(argv[5]);
  const std::string go_path = argv[6];
  const std::string out_dir = argv[7];
  const std::string child_binary = argv[8];
  const std::vector<std::string> child_migs(argv + 9, argv + argc);
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

  std::vector<llama_token> tokens = tokenize(vocab, read_file(prefix_path), true);
  mark = Clock::now();
  decode(context, tokens, 0, tokens.size(), batch);
  const double prefix_ms = ms_since(mark);
  const size_t prefix_tokens = tokens.size();

  mark = Clock::now();
  if (!llama_state_save_file(context, state_path.c_str(), tokens.data(),
                             tokens.size())) {
    throw std::runtime_error("cannot save the state " + state_path);
  }
  const double publish_ms = ms_since(mark);
  struct stat status {};
  const long long state_bytes = stat(state_path.c_str(), &status) == 0
                                    ? static_cast<long long>(status.st_size)
                                    : -1;
  std::printf("PUBLISHED prefix_tokens=%zu prefix_ms=%.1f publish_ms=%.2f "
              "state_bytes=%lld\n",
              prefix_tokens, prefix_ms, publish_ms, state_bytes);
  std::fflush(stdout);
  for (;;) {
    struct stat go {};
    if (stat(go_path.c_str(), &go) == 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // What a child needs beyond the state file.
  std::vector<std::pair<std::string, std::string>> changes;
  const char* exported = std::getenv("LLAMA_KV_VMM_EXPORT");
  const char* host = std::getenv("LLAMA_KV_HOST");
  if (exported != nullptr) {
    changes.emplace_back("LLAMA_KV_VMM_IMPORT", exported);
  } else if (host != nullptr) {
    changes.emplace_back("LLAMA_KV_PREFIX", std::to_string(prefix_tokens));
  }
  std::vector<pid_t> children;
  for (size_t index = 0; index < child_migs.size(); ++index) {
    auto environment = changes;
    environment.emplace_back("CUDA_VISIBLE_DEVICES", child_migs[index]);
    children.push_back(start_child(
        child_binary,
        {child_binary, "child", model_path, context_text, state_path,
         task_of(static_cast<int>(index)), generate_text},
        child_environment(environment),
        out_dir + "/child." + std::to_string(index)));
  }

  mark = Clock::now();
  const std::vector<llama_token> task = tokenize(
      vocab, "\n\nTask for the parent: list the kernel mechanisms named above.",
      false);
  tokens.insert(tokens.end(), task.begin(), task.end());
  decode(context, tokens, prefix_tokens, task.size(), batch);
  const double suffix_ms = ms_since(mark);

  mark = Clock::now();
  std::string text;
  for (int step = 0; step < n_gen; ++step) {
    const float* logits = llama_get_logits_ith(context, -1);
    llama_token token = static_cast<llama_token>(
        std::max_element(logits, logits + vocabulary) - logits);
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

  for (size_t index = 0; index < children.size(); ++index) {
    int child_status = 0;
    waitpid(children[index], &child_status, 0);
    std::printf("CHILD index=%zu exit=%d\n", index,
                WIFEXITED(child_status) ? WEXITSTATUS(child_status) : 128);
  }
  std::printf(
      "RESULT role=parent prefix_tokens=%zu model_ms=%.1f context_ms=%.1f "
      "suffix_ms=%.1f generation_ms=%.1f generated=%d generation_tps=%.3f "
      "total_ms=%.1f\n",
      prefix_tokens, model_ms, context_ms, suffix_ms, generation_ms, n_gen,
      generation_ms > 0.0 ? n_gen * 1000.0 / generation_ms : 0.0,
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
    std::fprintf(stderr, "kv_spawn: %s\n", error.what());
    return 1;
  }
}
