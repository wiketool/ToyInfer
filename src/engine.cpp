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
      logits_h(std::make_unique<float[]>(llm_config.vocab_size)),
      sampler(llm_config) {};

void Engine::chat() {
    uint32_t pos = 0;
    uint32_t token_id;
    uint32_t next_token_id;
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
        bool assistance_end = false;
        while (assistance_end == false && pos < options.max_seq_len) {
            if (pos < token_cnt) {
                token_id = token_ids[pos];
            } else {
                token_id = next_token_id;
            }
            transformer.forward(token_id, pos, logits_h);
            pos++;
            next_token_id = sampler.sample(logits_h);
            printf("next token id: %d\n", next_token_id);
            if (next_token_id == llm_config.eos_token_id) {
                assistance_end = true;
            }
            // printf("%s", tokenizer.decode(next_token_id));
        }
        printf("\n");

        linenoiseFree(line);
    }
}

}  // namespace toyinfer
