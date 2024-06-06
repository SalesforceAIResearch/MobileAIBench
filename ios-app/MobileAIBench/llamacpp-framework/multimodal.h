//
//  multimodal.h
//  llamacpp-framework
//
//  Created by Tulika Awalgaonkar and Sachin Desai on 6/3/24.
//

#import <Foundation/Foundation.h>

struct Options {
    uint32_t seed;              // RNG seed
    float   temperature;        // temperature
    int32_t n_gpu_layers;       // number of layers to store in VRAM (-1 - use default)
    int32_t n_ctx;              // context size
    int32_t n_batch;            // batch size for prompt processing
    int32_t n_predict;          // number of tokens to predict
    bool    embedding;          // compute embeddings
};

@interface MultiModal : NSObject

- (BOOL) isLoaded;

- (BOOL)loadModel:(NSString * _Nonnull)modelAtPath;

- (void)evaluateMultimodal:(NSString * _Nonnull)prompt
usingClipModelAtPath:(NSString * _Nonnull)clipPath
         modelAtPath:(NSString * _Nonnull)modelAtPath
        imageAtPaths:(NSArray * _Nonnull)imagePaths
          completion:(void (^ _Nonnull)(NSString * _Nullable value, BOOL isComplete, BOOL isError))completion;
@end
