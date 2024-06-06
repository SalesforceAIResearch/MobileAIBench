from modeling.model_configs.model_map import model_simple_name_map
from tasks.tasks2datasets import task2dataset, dataset2path
from data_processing.data_loader import Data
from modeling.model_wrapper import ModelWrapper
from evaluation.predict import ModelPredictor
from evaluation.evaluate import ModelEvaluator
import warnings
import json
import argparse
import json
import time
import pandas as pd
import shortuuid
import subprocess
from datetime import datetime
import os

def parse_arguments():
    """
    Parses and returns command-line arguments for the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(description="Evaluation pipeline for Mobile LLM models")
    
    # Grouping required arguments
    required_args = {
        '--task': {'choices': ['all', 'summarization', 'text2sql', 'question_answering', 'alpaca_eval', 'mt_bench', 'trust_and_safety', 'mmlu', 'math'], 'required': True, 'help': 'Evaluation task'},
        '--model_name': {'type': str, 'help': 'Path to the model file'},
        '--model_lib': {'choices': ['llama_cpp_python', 'huggingface', 'llama_cpp', 'openai'], 'default': 'llama_cpp_python', 'help': 'API for loading the model'}
    }
    
    # Grouping optional arguments
    optional_args = {
        '--log_output': {'action': 'store_true', 'help': 'Log model output'},
        '--data_path': {'help': 'Path to the data file'},
        '--use_gpu': {'action': 'store_true', 'help': 'Use GPU for inference'},
        '--timestamp': {'type': str, 'default': '', 'help': 'Timestamp for model'},
        '--mt_bench_path': {'type': str, 'default': 'FastChat/fastchat/llm_judge/data/mt_bench', 'help': 'path to mt_bench'},
        '--path_to_fastchat': {'type': str, 'default': '/export/home/code/MobileBench', 'help': 'path to mt_bench'},
        '--dataset_name': {'type': str, 'help': 'Name of the dataset', 'default':None},
        '--output_path': {'default': 'output/', 'help': 'Path to the output file'},
    }

    # task related arguments
    task_args = {
        '--mt_bench_num_choices': {'default': 1, 'type': int, 'help': 'mt_bench'},
    }

    for arg, kwargs in required_args.items():
        parser.add_argument(arg, **kwargs)
    for arg, kwargs in optional_args.items():
        parser.add_argument(arg, **kwargs)
    for arg, kwargs in task_args.items():
        parser.add_argument(arg, **kwargs)

    return parser.parse_args()


def setup_and_sanity_check(args):

    # setup arguments
    try:
        args.model_name_simplified = model_simple_name_map[args.model_name]
    except:
        raise ValueError("Model not registered. Please add the model details here: ./src/modeling/model_configs/")
        
    args.timestamp = str(time.time()) if args.timestamp == '' else args.timestamp
    if '.' in args.timestamp:
        args.timestamp = args.timestamp.split('.')[0]
    
    if args.task == 'all':
        args.task = ['summarization', 'text2sql', 'question_answering', 'alpaca_eval', 'mt_bench', 'trust_and_safety', 'mmlu', 'math']
    else:
        args.task = [args.task]

    os.makedirs(f"{args.output_path}alpaca_eval", exist_ok=True)
    args.alpaca_model_output = f"{args.output_path}alpaca_eval/{args.model_name_simplified}.json"
    args.mt_bench_model_output = f"{args.mt_bench_path}/model_answer/{args.model_name_simplified}_{args.timestamp}.jsonl"

    if args.dataset_name is None:
        args.dataset_name = {}
        for task in args.task:
            args.dataset_name[task] = task2dataset[task]
    else:
        assert args.dataset_name in task2dataset[args.task], "Incompatible task and dataset"

    # load model config
    with open(f"./src/modeling/model_configs/{args.model_name_simplified}/config.json", "r") as json_file:
        args.model_config = json.load(json_file)


    # sanity_check
    assert not (args.model_name == 'xgen-7b-instruct' and args.use_gpu), f"GPU inference not supported for {args.model_name} model."
    

def main():

    args = parse_arguments()
    setup_and_sanity_check(args)

    final_result = {}
    
    for task in args.task:
        final_result[task] = {}
        
        for dataset in args.dataset_name[task]:

            args.task_nm = task
            args.dataset_nm = dataset
            args.data_path = dataset2path[args.dataset_nm]
            
            data = Data(args)
            data.get_data()

            model = ModelWrapper(args)

            predictor = ModelPredictor(args, data, model)
            predictions = predictor.predict()  

            evaluator = ModelEvaluator(args, data, predictions)
            results = evaluator.evaluate()

            final_result[task][dataset] = results

            
    with open(args.output_path+model_simple_name_map[args.model_name]+'.txt', 'a+') as f:
        f.write('-'*50+'\n')
        f.write(str(args.model_name)+'\n'+str(args.timestamp)+'\n')
        f.write('-'*50+'\n')
        f.write(str(final_result)+'\n')
            

if __name__ == '__main__':
    main()