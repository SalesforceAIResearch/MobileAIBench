import numpy as np
import re
import subprocess
import time
import os
import json
import pandas as pd
import copy
import shortuuid
from tqdm import tqdm
import logging
from evaluation.utils import setup_log, setup_alpaca_log, mt_bench_temperature_config, reorg_answer_file
from modeling.model_configs.model_map import model_simple_name_map



class ModelPredictor():

    def __init__(self, args, data=None, model=None):
        self.args = args
        self.data = data
        self.model = model

    def clean_pred_summurization(self, pred):
        if ':' in pred and pred.index(':')<100:
            pred = pred.split(':', 1)[1].strip()
        return pred

    def clean_pred_text2sql(self, pred):
        if "```" in pred:
            try:
                pred = pred.replace("sql", "").replace('\n', ' ')
                match = re.search(r'```(.*?)```', pred)
                pred = match.group(1)
            except:
                pass
        return pred

    def clean_pred_QA(self, pred):
        if ':' in pred and pred.index(':')<100:
            pred = pred.split(':', 1)[1].strip()
        return pred

    def predict_summarization(self):
        predictions = []
            
        for prompt in tqdm(self.data.prompts_all):
            
            pred = self.model.predict(prompt)
            pred = self.clean_pred_summurization(pred)
            predictions.append(pred)

        if self.args.log_output:
            setup_log(self.args, self.data.prompts_all, predictions)

        return predictions

    def predict_text2sql(self):
        predictions = []

        for prompt in tqdm(self.data.prompts_all):
            
            pred = self.model.predict(prompt)
            pred = self.clean_pred_text2sql(pred)
            predictions.append(pred)

        if self.args.log_output:
            setup_log(self.args, self.data.prompts_all, predictions)

        return predictions

    def predict_question_answering(self):
        predictions = []

        for prompt in tqdm(self.data.prompts_all):
            
            pred = self.model.predict(prompt)
            pred = self.clean_pred_QA(pred)
            predictions.append(pred)

        if self.args.log_output:
            setup_log(self.args, self.data.prompts_all, predictions)

        return predictions
        

    def predict_alpaca_eval(self):
        predictions = []
        
        for idx, example in enumerate(tqdm(self.data.alpaca_eval_set)):
            temp = {"generator": self.args.model_name + '_' +self.args.timestamp, "instruction": example["instruction"]}
            temp["output"] = self.model.predict(example["hydrated_instruction"])
            predictions.append(temp)

        with open(self.args.alpaca_model_output, 'w') as file:
            json.dump(predictions, file)
        
        if self.args.log_output:
            setup_alpaca_log(self.args, predictions)

        return predictions

    def predict_mt_bench(self):

        for question in tqdm(self.data.mt_bench_questions):
            choices = []
            temperature = mt_bench_temperature_config.get(question["category"], 0.7)
            do_sample = False if temperature < 1e-4 else True
            
            for i in range(self.args.mt_bench_num_choices):
                turns = []
                conv = copy.deepcopy(self.args.model_config["conversation_template"])

                for qs in question["turns"]:
                    if len(conv["messages"]) == 0:
                        conv["messages"].append(conv["system_template"].format(
                            intro=conv["intro"],
                            system=conv["system"],
                            sep1=conv["sep1"],
                            prompt=qs,
                            sep2=conv["sep2"]
                        ))
                    else:
                        conv["messages"].append(conv["sep3"])
                        conv["messages"].append(qs)
                        conv["messages"].append(conv["sep2"])

                    prompt = "".join(conv["messages"])
                    pred = self.model.predict(prompt, temperature, do_sample)
                    conv["messages"].append(pred)
                    turns.append(pred)
                    
                choices.append({"index": i, "turns": turns})
                
            os.makedirs("/".join(self.args.mt_bench_model_output.split('/')[:-1]), exist_ok=True)
            with open(self.args.mt_bench_model_output, "a+") as fout:
                fout.write(json.dumps({
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": f"{self.args.model_name_simplified}_{self.args.timestamp}",
                    "choices": choices,
                    "tstamp": time.time(),
                }) + "\n")
        reorg_answer_file(self.args.mt_bench_model_output)

    def predict_trust_and_safety(self):

        predictions = []
        
        for prompt in tqdm(self.data.prompts_all):
            
            pred = self.model.predict(prompt)
            predictions.append(pred)
            

        # dumping predictions
        # TODO: Integrate Eval in MobileBench instead of handing over predictions

        fn = f"{self.args.dataset_nm}_{model_simple_name_map[self.args.model_name]}"
        data = {"system": self.data.system_ts, "prompt": self.data.prompt_ts, "answer": self.data.expected_ans_all, "prediction": predictions}
        # df = pd.DataFrame(data)
        # df.to_csv(f"/export/home/code/MobileBench/output/TS/{fn}.csv", mode='w', index=False)
        
        return predictions

    def predict_mmlu(self):

        predictions = []
        
        for prompt in tqdm(self.data.prompts_all):
            
            pred = self.model.predict(prompt)
            predictions.append(pred)
            

        # dumping predictions
        # TODO: Integrate Eval in MobileBench instead of handing over predictions

        fn = f"{self.args.dataset_nm}_{model_simple_name_map[self.args.model_name]}"
        data = {"system": self.data.system_ts, "prompt": self.data.prompt_ts, "answer": self.data.expected_ans_all, "prediction": predictions}
        # df = pd.DataFrame(data)
        # df.to_csv(f"/export/home/code/MobileBench/output/MMLU/{fn}.csv", mode='w', index=False)
        
        return predictions

    def predict_math(self):

        predictions = []
        
        for prompt in tqdm(self.data.prompts_all):
            
            pred = self.model.predict(prompt)
            predictions.append(pred)
            

        # dumping predictions
        # TODO: Integrate Eval in MobileBench instead of handing over predictions

        fn = f"{self.args.dataset_nm}_{model_simple_name_map[self.args.model_name]}"
        data = {"system": self.data.system_ts, "prompt": self.data.prompt_ts, "answer": self.data.expected_ans_all, "prediction": predictions}
        # df = pd.DataFrame(data)
        # df.to_csv(f"/export/home/code/MobileBench/output/MATH/{fn}.csv", mode='w', index=False)
        
        return predictions
        
        

    def predict(self):

        if self.args.task_nm == 'summarization':
            return self.predict_summarization()
        elif self.args.task_nm == 'text2sql':
            return self.predict_text2sql()
        elif self.args.task_nm == 'question_answering':
            return self.predict_question_answering()
        elif self.args.task_nm == 'alpaca_eval':
            return self.predict_alpaca_eval()
        elif self.args.task_nm == 'mt_bench':
            return self.predict_mt_bench()
        elif self.args.task_nm == 'trust_and_safety':
            return self.predict_trust_and_safety()
        elif self.args.task_nm == 'mmlu':
            return self.predict_mmlu()
        elif self.args.task_nm == 'math':
            return self.predict_math()
        else:
            raise NotImplementedError(f"{self.args.task_nm} not supported.")

    
            
    

            





            
