from evaluation.utils import normalize_answer, parse_sql_to_dict, grade_exact_match, grade_adv_instruction, grade_do_not_answer, grade_privacy_leakage
import collections
import string
import sqlparse
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from typing import List, Tuple, Union
from rouge_score import rouge_scorer
import numpy as np
import subprocess
import pandas as pd

class ModelEvaluator:
    def __init__(self, args, data, predictions):
        self.args = args
        self.data = data
        self.predictions = predictions

    def evaluate(self):
        
        if self.args.task_nm == 'summarization':
            return self.evaluate_summarization()
        elif self.args.task_nm == 'text2sql':
            return self.evaluate_text2sql()
        elif self.args.task_nm == 'question_answering':
            return self.evaluate_question_answering()
        elif self.args.task_nm == 'alpaca_eval':
            return self.evaluate_alpaca_eval()
        elif self.args.task_nm == 'mt_bench':
            return self.evaluate_mt_bench()
        elif self.args.task_nm == 'trust_and_safety':
            return self.evaluate_trust_and_safety()
        elif self.args.task_nm == 'mmlu':
            return self.evaluate_mmlu()
        elif self.args.task_nm == 'math':
            return self.evaluate_math()
        else:
            raise NotImplementedError(f"{self.args.task_nm} not supported.")

    
    def evaluate_mmlu(self):
        score = grade_exact_match(self.data.prompt_ts, self.data.expected_ans_all, self.data.predictions)
        return {f"llm-judge-score-for-{self.args.dataset_nm}": score}

    def evaluate_math(self):
        score = grade_exact_match(self.data.prompt_ts, self.data.expected_ans_all, self.data.predictions)
        return {f"llm-judge-score-for-{self.args.dataset_nm}": score}
        
        
    def evaluate_trust_and_safety(self):
        
        if self.args.dataset_nm in ['bbq', 'social_chemistry_101', 'truthful_qa']:
            score = grade_exact_match(self.data.prompt_ts, self.data.expected_ans_all, self.data.predictions)
        elif self.args.dataset_nm == 'adv_instruction':
            score = grade_adv_instruction(self.data.prompt_ts, self.data.expected_ans_all, self.data.predictions)
        elif self.args.dataset_nm == 'do_not_answer':
            score = grade_do_not_answer(self.data.prompt_ts, self.data.expected_ans_all, self.data.predictions)
        elif self.args.dataset_nm == 'privacy_leakage':
            score = grade_privacy_leakage(self.data.prompt_ts, self.data.expected_ans_all, self.data.predictions)
        else:
            raise NotImplementedError(f"{self.args.dataset_nm} not supported.")

        return {f"llm-judge-score-for-{self.args.dataset_nm}": score}

    

    def evaluate_summarization(self):

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(exp, pred) for pred, exp in zip(self.predictions, self.data.expected_ans_all)]
        r1 = [score['rouge1'].fmeasure for score in scores]
        rl = [score['rougeL'].fmeasure for score in scores]
        
        return {"rouge1": np.mean(r1), "rougeL": np.mean(rl)}

    def compare_sql_queries(self, query1, query2):
        """
        Compare the similarity of two SQL queries using the Levenshtein ratio.
        
        Parameters:
            query1 (str): The first SQL query.
            query2 (str): The second SQL query.
    
        Returns:
            float: The similarity ratio.
        """
        return Levenshtein.ratio(query1, query2)
    
    def compute_levenshtein_score(self, predicted_output, actual_output):
        """
        Compute the Levenshtein distance score between two text strings.
        
        Parameters:
            predicted_output (str): The predicted output
            actual_output (str): The actual output
    
        Returns:
            float: The Levenshtein score (ratio) indicating the similarity between the predicted and actual output.
        """
        return self.compare_sql_queries(predicted_output, actual_output)

    def calculate_f1_score(self, l1, l2):
        """
        Calculate the F1 score based on two lists of elements.
        
        Parameters:
            l1 (List[str]): First list of elements.
            l2 (List[str]): Second list of elements.
    
        Returns:
            float: The F1 score.
        """
        # Calculate the true positives, false positives, and false negatives
        tp = len(set(l1) & set(l2))
        fp = len(set(l1) - set(l2))
        fn = len(set(l2) - set(l1))
    
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    
        return f1_score

    def compute_sqlparser_score(self, predicted_output, actual_output):
        """
        Compute the SQLParser F1 score for a predicted and actual SQL query.
    
        Parameters:
            predicted_output (str): The predicted SQL query.
            actual_output (str): The actual SQL query.
    
        Returns:
            float: The F1 score based on SQL parsing.
        """
    
        gt = parse_sql_to_dict(actual_output)
        pred = parse_sql_to_dict(predicted_output)
        gt = [(x[0], str(x[1]).lower()) for x in gt]
        pred = [(x[0], str(x[1]).lower()) for x in pred]
    
        return self.calculate_f1_score(gt, pred)

    def evaluate_text2sql(self):
        
        sqlparser_scores = [self.compute_sqlparser_score(pred, ans) for pred, ans in zip(self.predictions, self.data.expected_ans_all)]
        levenshtein_scores = [self.compute_levenshtein_score(pred, ans) for pred, ans in zip(self.predictions, self.data.expected_ans_all)]
        return {"SQLParser": np.mean(sqlparser_scores), "Levenshtein": np.mean(levenshtein_scores)}

    def compute_exact_match(self, prediction, ground_truth):
        """
        Compute if the normalized prediction matches the normalized ground truth exactly.
        
        Parameters:
            prediction (str): The predicted answer.
            ground_truth (str): The actual answer.
    
        Returns:
            int: 1 if matches exactly, otherwise 0.
        """
        return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0

    def compute_f1(self, prediction, ground_truth):
        """
        Compute the F1 score, precision, and recall for a single prediction against
        the ground truth.
    
        Parameters:
            prediction (str): The predicted text.
            ground_truth (str): The actual text.
    
        Returns:
            Tuple[float, float, float]: The F1 score, precision, and recall.
        """
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0, 0, 0
        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def compute_BLEU(self, expected_tokens, actual_tokens):
        """
        Compute the BLEU score for a pair of expected and actual text tokens.
        
        Parameters:
            expected_tokens (str): The expected text.
            actual_tokens (str): The actual text.
    
        Returns:
            float: The BLEU score.
        """
        expected_tokens=normalize_answer(expected_tokens)
        actual_tokens=normalize_answer(actual_tokens)
        bleu_score = sentence_bleu(
            [expected_tokens], actual_tokens, 
            smoothing_function=SmoothingFunction().method1)
        return bleu_score

    def evaluate_question_answering(self):
        """
        Computes Exact Match, F1, and BLEU scores for question answering tasks.
        """
        em_scores = [self.compute_exact_match(pred, ans) for pred, ans in zip(self.predictions, self.data.expected_ans_all)]
        f1_scores = [self.compute_f1(pred, ans)[0] for pred, ans in zip(self.predictions, self.data.expected_ans_all)]
        bleu_scores = [self.compute_BLEU(pred, ans) for pred, ans in zip(self.predictions, self.data.expected_ans_all)]
        return {"EM": np.mean(em_scores), "F1": np.mean(f1_scores), "BLEU": np.mean(bleu_scores)}

    def evaluate_alpaca_eval(self):
        """
        Execute the Alpaca evaluation tool for the model outputs.
        """
        tmp_nm = self.args.model_name + '_' + self.args.timestamp
        command = f"alpaca_eval --model_outputs {self.args.alpaca_model_output}"
        subprocess.run(command, shell=True)
        df = pd.read_csv(f'{self.args.output_path}/alpaca_eval/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv')
        return list(df[df['Unnamed: 0'] == f"{tmp_nm}"][['length_controlled_winrate']].items())
    

    def evaluate_mt_bench(self):
        """
        Computes and returns the MT-Bench evaluation metrics for the model outputs.
    
        Returns:
            dict: A dictionary containing the mean score for each evaluated model.
        """
        tmp_nm = f"{self.args.model_name_simplified}_{self.args.timestamp}"
        eval_command = f"python {self.args.path_to_fastchat}/FastChat/fastchat/llm_judge/gen_judgment.py --model-list {tmp_nm} --parallel 4"
        subprocess.run(eval_command, shell=True)

        df_all = pd.read_json(f"{self.args.mt_bench_path}/model_judgment/gpt-4_single.jsonl", lines=True)
        df = df_all[df_all["score"] != -1]
        results = {"results": df[["model", "score"]].groupby(["model"]).mean()['score'].to_dict()}
        return results
