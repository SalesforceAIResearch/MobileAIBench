//
//  multimodal.mm
//  llamacpp-framework
//
//  Created by Tulika Awalgaonkar and Sachin Desai on 6/3/24.
//

#import "multimodal.h"
#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "log.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"
#include "llava_cli.h"


typedef struct ModelState {
    gpt_params params;
    llama_model *model;
    llava_context *llava_ctx;
    bool loaded;
    struct Options options =  { (uint32_t)-1, 0.1, 32, 512, 512, -1, false};
} ModelState;


class finally {
    std::function<void(void)> functor;
public:
    finally(const std::function<void(void)> &functor) : functor(functor) {}
    ~finally() {
        functor();
    }
};

@implementation MultiModal {
    ModelState state;
}

- (BOOL) isLoaded {
    return state.loaded;
}

- (void)setParamValues {
    state.params.seed         = state.options.seed;
    state.params.n_gpu_layers = state.options.n_gpu_layers;
    state.params.n_ctx        = state.options.n_ctx;
    state.params.n_batch      = state.options.n_batch;
    state.params.n_predict    = state.options.n_predict;
    state.params.sparams.temp = state.options.temperature;
    state.params.n_threads = std::thread::hardware_concurrency();
    if (state.params.n_threads_batch <= 0) {
        state.params.n_threads_batch = std::thread::hardware_concurrency();
    }
    state.params.embedding    = state.options.embedding;
}

- (BOOL)loadModel:(NSString *)modelAtPath{
    [self setParamValues];

    state.params.n_gpu_layers = 999;
    state.params.n_ctx = 4096;
    state.params.model = [modelAtPath cStringUsingEncoding:NSUTF8StringEncoding];
    state.model = llava_init(&state.params);
    
    state.loaded=true;
    return true;
}

- (void)evaluateMultimodal:(NSString * _Nonnull)prompt
usingClipModelAtPath:(NSString * _Nonnull)clipPath
     modelAtPath:(NSString * _Nonnull)modelAtPath
     imageAtPaths:(NSArray * _Nonnull)imagePaths
      completion:(void (^ _Nonnull)(NSString * _Nullable value, BOOL isComplete, BOOL isError))completion {

    state.params.mmproj = [clipPath cStringUsingEncoding:NSUTF8StringEncoding];

    for (NSString *imagePath: imagePaths) {
        state.params.image.push_back([imagePath cStringUsingEncoding:NSUTF8StringEncoding]);
    }
    state.params.prompt = [prompt cStringUsingEncoding:NSUTF8StringEncoding];

    printf("model: %s\n", state.params.model.c_str());
    printf("projector: %s\n", state.params.mmproj.c_str());

    if (state.params.mmproj.empty() || (state.params.image.empty() && !prompt_contains_image(state.params.prompt))) {
        if (completion) {
            completion(@"no clip model or image", true, true);
        }
        return;
    }
    
    if (![self isLoaded]){
        if (![self loadModel:modelAtPath]) {
            if (completion) {
                completion(@"failed to init llava model", true, true);
            }
            return;
        }
    }
    
    
    std::function<void(const std::string &, bool, bool)> cppLambda = [completion](const std::string &str, bool isComplete, bool isError) {
        NSString *nsStr = [NSString stringWithUTF8String:str.c_str()];
        completion(nsStr, isComplete, isError);
    };

    for (auto &image: state.params.image) {
        state.llava_ctx = llava_init_context(&state.params, state.model);
        auto image_embed = load_image(state.llava_ctx, &state.params, image);
        if (!image_embed) {
            completion(@"failed to load image", true, true);
            return;
        }

        // process the prompt
        process_prompt(state.llava_ctx, image_embed, &state.params, state.params.prompt, cppLambda);

        llama_print_timings(state.llava_ctx->ctx_llama);
        llava_image_embed_free(image_embed);
        llava_free(state.llava_ctx);
    }

    //llama_free_model(model);
}


@end
