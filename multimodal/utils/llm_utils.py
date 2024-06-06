from llama_cpp import Llama
from utils.utils import load_config
from llama_cpp.llama_chat_format import Llava15ChatHandler, MoondreamChatHandler


def init_llm(seed, args):
    model_config = load_config(args.model, args.model_config_path)
    if args.multimodal:
        llm = init_llm_llama_cpp_multimodal(seed, model_config)
    else:
        llm = init_llm_llama_cpp(seed, model_config)
    return llm


def init_llm_llama_cpp(seed, model_config):
    llm = Llama(
        model_path=model_config['model_path'],
        seed=seed,
        n_ctx=model_config['context_length'])
    return llm


def init_llm_llama_cpp_multimodal(seed, model_config):
    if "moondream" in model_config['name']:
        chat_handler = MoondreamChatHandler(clip_model_path=model_config['chat_handler_path'])
    else:
        chat_handler = Llava15ChatHandler(clip_model_path=model_config['chat_handler_path'])
    llm = Llama(
        model_path=model_config['model_path'],
        chat_handler=chat_handler,
        seed=seed,
        n_ctx=model_config['context_length'],
        logits_all=True,
        )
    return llm