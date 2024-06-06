import pandas as pd
import datasets
from data_processing.utils import load_questions

class Data():

    def __init__(self, args):
        self.args = args
        self.prompts_all = []
        self.expected_ans_all = []
        self.system_ts, self.prompt_ts = [], []
        self.alpaca_eval_set = None
        self.mt_bench_questions = None
        self.prompt_format = args.model_config["prompt_template"]

    def get_data(self):

        if self.args.task_nm == 'summarization':
            self.load_summarization_data()
        elif  self.args.task_nm == 'text2sql':
            self.load_text2sql_data()
        elif  self.args.task_nm == 'question_answering':
            self.load_question_answering_data()
        elif  self.args.task_nm == 'alpaca_eval':
            self.load_alpaca_eval_data()
        elif  self.args.task_nm == 'mt_bench':
            self.load_mt_bench_data()
        elif self.args.task_nm == 'trust_and_safety':
            self.load_trust_and_safety_data()
        elif self.args.task_nm == 'mmlu':
            self.load_mmlu_data()
        elif self.args.task_nm == 'math':
            self.load_math_data()
        else:
            raise ValueError(f"Task name `{self.args.task_nm}` is invalid.")

    def load_mmlu_data(self):
        df = pd.read_csv(self.args.data_path)
        
        for sys, pro, ans in zip(df['system'], df['prompt'], df['answer']):
            if len(str(sys)) > 10: 
                prompt = self.prompt_format.replace("{system}", sys).replace("{prompt}", pro)
            else:
                prompt = self.prompt_format.replace("{system}", '').replace("{prompt}", pro)
            self.prompts_all.append(prompt)
            self.expected_ans_all.append(ans)
            self.system_ts.append(sys)
            self.prompt_ts.append(pro)

    def load_math_data(self):
        df = pd.read_csv(self.args.data_path)
        
        for sys, pro, ans in zip(df['system'], df['prompt'], df['answer']):
            if len(str(sys)) > 10: 
                prompt = self.prompt_format.replace("{system}", sys).replace("{prompt}", pro)
            else:
                prompt = self.prompt_format.replace("{system}", '').replace("{prompt}", pro)
            self.prompts_all.append(prompt)
            self.expected_ans_all.append(ans)
            self.system_ts.append(sys)
            self.prompt_ts.append(pro)
        

    def load_trust_and_safety_data(self):
        df = pd.read_csv(self.args.data_path)
        
        for sys, pro, ans in zip(df['system'], df['prompt'], df['answer']):
            if len(str(sys)) > 10: 
                prompt = self.prompt_format.replace("{system}", sys).replace("{prompt}", pro)
            else:
                prompt = self.prompt_format.replace("{system}", '').replace("{prompt}", pro)
            self.prompts_all.append(prompt)
            self.expected_ans_all.append(ans)
            self.system_ts.append(sys)
            self.prompt_ts.append(pro)
            

    def load_summarization_data(self):

        df = pd.read_csv(self.args.data_path)

        article_key = 'article' if 'article' in df.columns else 'document'
        summary_key = 'highlights' if 'highlights' in df.columns else 'summary'
        SYS = "You're a helpful assistant who is good at summarizing articles."
        for context, answer in zip(df[article_key], df[summary_key]):
            QUESTION = "Create a short summary of the following article: "+context.replace("'", "\\'").replace('"', '\\"')+"\n"
            prompt = self.prompt_format.replace("{system}", SYS).replace("{prompt}", QUESTION)
            self.prompts_all.append(prompt)
            self.expected_ans_all.append(answer)
        # self.prompts_all = self.prompts_all[:10]
        # self.expected_ans_all = self.expected_ans_all[:10]
       
            

    def load_text2sql_data(self):

        df = pd.read_csv(self.args.data_path)

        contexts = df['context'].tolist()
        questions = df['question'].tolist()
        answers = df['answer'].tolist()
        SYS = "You're a helpful assistant proficient in crafting SQL queries."
    
        for context, question, answer in zip(contexts, questions, answers):
            CONTEXT = f"The following command was used to create the SQL table: {context}\n"
            QUESTION = f"question: write a SQL query based on hte provided table for the following inquiry: {question}\nanswer: "
            prompt = self.prompt_format.replace("{system}", SYS).replace("{prompt}", CONTEXT+QUESTION)
            self.prompts_all.append(prompt)
            self.expected_ans_all.append(answer)
        # self.prompts_all = self.prompts_all[:10]
        # self.expected_ans_all = self.expected_ans_all[:10]


    def load_question_answering_data(self):

        df = pd.read_csv(self.args.data_path)

        question_key = 'question' if self.args.dataset_nm == 'hotpot_qa' else 'instruction'
        answer_key = 'answer' if self.args.dataset_nm == 'hotpot_qa' else 'response'
        context_key = 'context' 
    
        # System prompt differs by dataset, but the template is consistent
        SYS = "You're a helpful assistant proficient in answering questions based on the provided context. Directly provide the final answer without any reasoning or justification."
    
        for index, row in df.iterrows():
            context = row.get(context_key, '') if self.args.model_config[self.args.model_lib]["basic"]["question_answering"]["use_context"] else ''
            question = row[question_key]
            answer = row[answer_key]                
    
            # Constructing the prompt with or without context based on the dataset and configuration
            CONTEXT_PREFIX = f"context: {context}\n" if self.args.model_config[self.args.model_lib]["basic"]["question_answering"]["use_context"] and context else ""
            QUESTION = f"question: {question}\nanswer: "
            
            prompt = self.prompt_format.replace("{system}", SYS).replace("{prompt}", CONTEXT_PREFIX+QUESTION)
            self.prompts_all.append(prompt)
            self.expected_ans_all.append(answer)  
        # self.prompts_all = self.prompts_all[:10]
        # self.expected_ans_all = self.expected_ans_all[:10]


    def load_alpaca_eval_data(self):

        SYS = "You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

        def update_instruction(example):
            example["hydrated_instruction"] = self.prompt_format.replace("{system}", SYS).replace("{prompt}", example["instruction"])
            return example
        self.alpaca_eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        self.alpaca_eval_set = self.alpaca_eval_set.map(update_instruction)


    def load_mt_bench_data(self):
        self.mt_bench_questions = load_questions(self.args.data_path)


        
        


        
        