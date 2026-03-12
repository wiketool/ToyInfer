#include "engine.h"

#include <cstring>
#include <memory>

#include "options.h"
#include "tokenizer.h"
#include "transformer.h"

namespace toyinfer {
Engine::Engine(const Options& options)
    : options(options),
      llm_config(options),
      tokenizer(options, llm_config),
      transformer(options, llm_config),
      logits_h(std::make_unique<float[]>(llm_config.vocab_size)) {};

void Engine::chat() {
    while (1) {
        char* line = linenoise("ToyInfer> ");
        if (line == nullptr) {
            continue;
        }
        if (strcmp(line, "\\quit") == 0) {
            break;
        }
        std::unique_ptr<uint32_t[]> token_ids;
        uint32_t token_cnt;
        std::unique_ptr<char[]> prompt;
        tokenizer.render_prompt(prompt, line, nullptr);
        std::cout << prompt.get() << std::endl;
        tokenizer.encode(prompt.get(), token_ids, token_cnt);
        for (uint32_t i = 0; i < token_cnt; i++) {
            std::cout << token_ids[i] << " ";
        }
        std::cout << std::endl;
        
        linenoiseFree(line);
    }
}

}  // namespace toyinfer
