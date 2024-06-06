//
//  llava_cli.h
//  llamacpp-framework
//
//  Created by Tulika Awalgaonkar and Sachin Desai on 6/3/24.
//

#ifndef llava_cli_h
#define llava_cli_h

#include "ggml.h"


#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAVA_API __declspec(dllexport)
#        else
#            define LLAVA_API __declspec(dllimport)
#        endif
#    else
#        define LLAVA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAVA_API
#endif


#ifdef __cplusplus
extern "C" {
#endif

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

LLAVA_API bool prompt_contains_image(const std::string& prompt);
LLAVA_API struct llama_model *llava_init(gpt_params *params);
LLAVA_API struct llava_context *llava_init_context(gpt_params *params, llama_model *model);
LLAVA_API struct llava_image_embed * load_image(llava_context *ctx_llava, gpt_params *params, const std::string &fname);
LLAVA_API void process_prompt(struct llava_context *ctx_llava, struct llava_image_embed *image_embed, gpt_params *params, const std::string &prompt, std::function<void(const std::string&, bool, bool)> cppLambda);
LLAVA_API void llava_free(struct llava_context *ctx_llava);


LLAVA_API

#ifdef __cplusplus
}
#endif

#endif /* llava_cli_h */
