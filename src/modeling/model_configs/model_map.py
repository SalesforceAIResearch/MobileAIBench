import torch 

model_simple_name_map = {
    "google/gemma-2b-it": "gemma-2b-it",
    "meta-llama/Llama-2-7b-chat-hf": "llama2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b-it",
    "google/gemma-7b-it": "gemma-7b-it",
    "microsoft/phi-2": "phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama-1b-chat",
    "stabilityai/stablelm-zephyr-3b": "zephyr-3b",
    "Salesforce/xgen2": "xgen-3b-v3",
    "xgen1": "xgen-3b-v3",
    "xgen2": "xgen-3b-v3",
    "gpt-4": "gpt-4",

    "/export/share/rmurthy/mobile_llm_models/8bit/gemma-2b-it.gguf": "gemma-2b-it",
    "/export/share/rmurthy/mobile_llm_models/8bit/llama2-7b-chat.gguf": "llama2-7b-chat-hf",
    "/export/share/rmurthy/mobile_llm_models/8bit/mistral-7b-instruct.gguf": "mistral-7b-it",
    "/export/share/rmurthy/mobile_llm_models/8bit/gemma-7b-it.gguf": "gemma-7b-it",
    "/export/share/rmurthy/mobile_llm_models/8bit/phi2-2.7b.gguf": "phi-2",
    "/export/share/rmurthy/mobile_llm_models/8bit/tinyllama-1.1b-chat.gguf": "tinyllama-1b-chat",
    "/export/share/rmurthy/mobile_llm_models/8bit/stablelm-zephyr-3b.gguf": "zephyr-3b",

    "/export/share/rmurthy/mobile_llm_models/4bit/gemma-2b-it.gguf": "gemma-2b-it",
    "/export/share/rmurthy/mobile_llm_models/4bit/llama2-7b-chat.gguf": "llama2-7b-chat-hf",
    "/export/share/rmurthy/mobile_llm_models/4bit/mistral-7b-instruct.gguf": "mistral-7b-it",
    "/export/share/rmurthy/mobile_llm_models/4bit/gemma-7b-it.gguf": "gemma-7b-it",
    "/export/share/rmurthy/mobile_llm_models/4bit/phi2-2.7b.gguf": "phi-2",
    "/export/share/rmurthy/mobile_llm_models/4bit/tinyllama-1.1b-chat.gguf": "tinyllama-1b-chat",
    "/export/share/rmurthy/mobile_llm_models/4bit/stablelm-zephyr-3b.gguf": "zephyr-3b",

    "/export/share/liangweiyang/Models/F16/google-gemma-2b-it": "gemma-2b-it",
    "/export/share/liangweiyang/Models/F16/llama-2-7b-chat-hf-f16.gguf": "llama2-7b-chat-hf",
    "/export/share/liangweiyang/Models/F16/mistral-7b-instruct": "mistral-7b-it",
    "/export/share/liangweiyang/Models/F16/google-gemma-7b-it": "gemma-7b-it",
    "/export/share/liangweiyang/Models/F16/phi2-2.7b": "phi-2",
    "/export/share/liangweiyang/Models/F16/tinyllama-1.1b-chat": "tinyllama-1b-chat",
    "/export/share/liangweiyang/Models/F16/stablelm-zephyr-3b": "zephyr-3b",

    "/export/share/liangweiyang/Models/Q4_0/gemma-2b-it": "gemma-2b-it",
    "/export/share/liangweiyang/Models/Q4_0/llama2-7b-chat": "llama2-7b-chat-hf",
    "/export/share/liangweiyang/Models/Q4_0/mistral-7b-instruct": "mistral-7b-it",
    "/export/share/liangweiyang/Models/Q4_0/phi2-2.7b": "phi-2",
    "/export/share/liangweiyang/Models/Q4_0/tinyllama-1.1b-chat": "tinyllama-1b-chat",
    "/export/share/liangweiyang/Models/Q4_0/stablelm-zephyr-3b": "zephyr-3b",
    "/export/share/liangweiyang/Models/Q4_0/gemma-7b-it_Q4_0.gguf": "gemma-7b-it",

    "/export/share/liangweiyang/Models/Q4_0/gemma-2b-it_new.gguf": "gemma-2b-it",
    "/export/share/liangweiyang/Models/F16/gemma-2b-it_new.gguf": "gemma-2b-it",
    "/export/share/liangweiyang/Models/F16/gemma-7b-it_f16.gguf": "gemma-7b-it",
    
}

model_dtype_map = {
    'zephyr-3b': torch.bfloat16,
    'mistral-7b-it': torch.bfloat16,
    'gemma-2b-it': torch.bfloat16,
    'gemma-7b-it': torch.bfloat16,
    'tinyllama-1b-chat': torch.bfloat16,
    'xgen-3b-v3': torch.bfloat16,
    'llama2-7b-chat-hf': torch.float16,
    'phi-2': torch.float16
}