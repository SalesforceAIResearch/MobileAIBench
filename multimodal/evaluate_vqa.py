import argparse
import random
import time
import tqdm
import os
import json
from pathlib import Path
# from datasets import list_metrics
from utils.llm_utils import init_llm
# from utils.utils import load_config
from utils.data_utils import data_processing

# from urllib.error import URLError, HTTPError


def generate_prompt(suffix_, question_content):
    prompt = question_content + "\n" + suffix_
    prompt = prompt.strip()
    return prompt

def main(args):
    if args.seed is None:
        seed = random.randint()
    else:
        seed = args.seed

    log_path = os.path.join(args.log_dir, args.dataset, args.model)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    llm = init_llm(seed, args)

    # test load dataset
    questions, answers, images = data_processing(dataset=args.dataset, 
                                   downsample_size=args.down_sample,
                                   dataset_dir=args.dataset_dir, 
                                   split="test")

    suffixes = {}
    with open('suffixes.txt', 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            suffixes[key] = value

    gts = []
    lines = []
    for q_id, q_dict in tqdm.tqdm(list(questions.items())):
        image_id = q_dict['image_id']
        question_content = q_dict['question']
        image_uri = images[image_id]
        if args.dataset == "vqav2":
            prompt = generate_prompt(suffixes["SUFFIX_VQAV2"], question_content)
        elif args.dataset == "viswiz":
            prompt = generate_prompt(suffixes["SUFFIX_VISWIZ"], question_content)
        elif args.dataset == "scienceqa":
            prompt = generate_prompt(suffixes["SUFFIX_SCIENCEQA"], question_content)
        elif args.dataset == "textvqa":
            prompt = generate_prompt(suffixes["SUFFIX_TEXTVQA"], question_content)
        elif args.dataset == "gqa":
            prompt = generate_prompt(suffixes["SUFFIX_GQA"], question_content)
        else:
            raise ValueError(f"Suffix for dataset {args.dataset} not specified.")

        try:
            start_time = time.time()
            output = llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": "You are an assistant for Visual Question Answering."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type" : "text", "text": prompt}
                        ]
                    }
                ],
                stop=["\n"]
            )
            gts.append({'question_id': q_id, "gt": answers[q_id]})
            latency = time.time() - start_time
            prediction = output['choices'][0]['message']['content']
            input_token_num = output['usage']["prompt_tokens"]
            output_token_num = output['usage']["completion_tokens"]
            total_tokens = output['usage']["total_tokens"]
            if args.dataset == "vqav2":
                lines.append(f"Question ID: {q_id}, Prediction: {prediction}, Gt: {answers[q_id]['multiple_choice_answer']}, Latency: {latency:.4f}, Input: {input_token_num:.4f}, Output: {output_token_num:.4f}, Total: {total_tokens:.4f}\n")
            elif args.dataset == "viswiz":
                lines.append(f"Question ID: {q_id}, Prediction: {prediction}, Gt: {answers[q_id]['answers'][0]['answer']}, Latency: {latency:.4f}, Input: {input_token_num:.4f}, Output: {output_token_num:.4f}, Total: {total_tokens:.4f}\n")
            elif args.dataset == "scienceqa":
                lines.append(f"Question ID: {q_id}, Prediction: {prediction}, Gt: {answers[q_id]['answer']}, Latency: {latency:.4f}, Input: {input_token_num:.4f}, Output: {output_token_num:.4f}, Total: {total_tokens:.4f}\n")
            elif args.dataset == "textvqa":
                lines.append(f"Question ID: {q_id}, Prediction: {prediction}, Gt: {answers[q_id]['answer'][0]}, Latency: {latency:.4f}, Input: {input_token_num:.4f}, Output: {output_token_num:.4f}, Total: {total_tokens:.4f}\n")
            elif args.dataset == "gqa":
                lines.append(f"Question ID: {q_id}, Prediction: {prediction}, Gt: {answers[q_id]['answer']}, Latency: {latency:.4f}, Input: {input_token_num:.4f}, Output: {output_token_num:.4f}, Total: {total_tokens:.4f}\n")
        except Exception as e:
            print(f"Unexpected error for question {q_id}")

    with open(os.path.join(log_path, f"eval_{args.down_sample}_seed_{args.seed}.txt"), 'w') as f:
        f.writelines(lines)
    with open(os.path.join(log_path, f"eval_{args.down_sample}_seed_{args.seed}_gt.json"), "w") as f:
        json.dump(gts, f, indent=4)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str, help="dataset name, now support [vqav2, viswiz, gqa, textvqa, scienceqa]")
    parser.add_argument("--model_config_path", type=str, default="mobile_lm/config/model_config.yaml")
    parser.add_argument("--down_sample", type=int, default=500, help="downsample size")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--task", type=str, default="qa", help="evaluation task")
    parser.add_argument("--log_dir", type=str,default="logs")
    parser.add_argument("--dataset_dir", type=str, default="/export/home/datasets/", help="dataset directory")
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    main(args)