# from llama_cpp import Llama
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from modeling.model_configs.model_map import model_dtype_map


# class ModelWrapper():
#     def __init__(self, args):
        
#         self.config = args.model_config
#         self.task_nm = args.task_nm
#         self.model_lib = args.model_lib
#         self.model_name = args.model_name

#         if args.model_lib == 'llama_cpp_python':
#             gpu_flag = -1 if args.use_gpu else 0
#             self.model = Llama(model_path=args.model_name, n_gpu_layers=gpu_flag, n_ctx=self.config[args.model_lib]["basic"]["n_ctx"])
#         # elif args.model_lib == 'llama_cpp':
#         #     self.model = None
#         elif args.model_lib == 'huggingface':
#             self.device = "cuda" if args.use_gpu else "cpu"
#             dtype = model_dtype_map[args.model_name_simplified]
            
#             if 'xgen1' in args.model_name.lower():
#                 self.model = AutoModelForCausalLM.from_pretrained("/export/share/bpang/pytorch_ckpts/exp_552_xgen3b_2624", ignore_mismatched_sizes=True, torch_dtype=dtype).to(self.device)
#                 self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen2", trust_remote_code=True, torch_dtype=dtype)
#             elif 'xgen2' in args.model_name.lower():
#                 self.model = AutoModelForCausalLM.from_pretrained("/export/share/bpang/pytorch_ckpts/exp_901_xgen1b_5248", ignore_mismatched_sizes=True, torch_dtype=dtype).to(self.device)
#                 self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen2", trust_remote_code=True, torch_dtype=dtype)
#             else:
#                 self.model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=dtype).to(self.device)
#                 self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=dtype)
                
#         else:
#             raise NotImplementedError("Library not supported.")
        

#     def predict(self, prompt, temperature_explicit=None, do_sample_explicit=None):
        
#         kwargs = self.config[self.model_lib][self.task_nm]
#         if temperature_explicit:
#             kwargs["temperature"] = temperature_explicit
#         if "do_sample" in kwargs and do_sample_explicit is not None:
#             kwargs["do_sample"] = do_sample_explicit

#         if self.model_lib == 'llama_cpp_python':
#             # tokens = self.model.tokenize(prompt.encode())
#             # tokens = tokens[:3900] if len(tokens) > 4000 else tokens
#             # pred = []
#             # completion_tokens = []
#             # for token in self.model.generate(tokens, temp=kwargs["temperature"], repeat_penalty=kwargs["repeat_penalty"], penalize_nl=kwargs["penalize_nl"]):
#             #     completion_tokens.append(token)
#             # op = self.model.detokenize(completion_tokens[:kwargs["max_tokens"]], prev_tokens=tokens)            
#             # pred = op.decode("utf-8")
#             # print(pred)
            
#             try:
#                 pred = self.model(prompt, **kwargs)['choices'][0]['text']
#             except:
#                 pred = self.model(prompt[:4000], **kwargs)['choices'][0]['text']

#         # elif self.model_lib == 'llama_cpp':
#         #     import subprocess
#         #     prompt = prompt.replace('"', '\"').replace("'", "\'").replace("(", "\(").replace(")", "\)")
#         #     print('************************ENTRY\n')
#         #     import os

#         #     current_file_path = os.path.abspath(__file__)
#         #     print("Current file path:", current_file_path)
#         #     with open('./abc.txt', 'w') as f:
#         #         f.write(prompt)
#         #     t = kwargs["temperature"]
#         #     mt = kwargs["max_tokens"]
#         #     xyz = f'./llama.cpp/main -m {self.model_name} -p "{prompt}" -e --temp {t} -n {mt} -c 4096 -ngl 35'
#         #     print(xyz)
#         #     print('*'*50)
#         #     process = subprocess.Popen([f'{xyz}'], stdout=subprocess.PIPE, shell=True)
#         #     pred = process.stdout.read().decode("utf-8")
#         #     print(pred)
#         #     import sys
#         #     sys.exit()
            
            
#         elif self.model_lib == 'huggingface':
#             input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
#             output = self.model.generate(input_ids, **kwargs)
#             pred = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
#         return pred


from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from modeling.model_configs.model_map import model_dtype_map
from openai import OpenAI


class ModelWrapper():
    def __init__(self, args):
        
        self.config = args.model_config
        self.task_nm = args.task_nm
        self.model_lib = args.model_lib

        if args.model_lib == 'llama_cpp_python':
            gpu_flag = -1 if args.use_gpu else 0
            self.model = Llama(model_path=args.model_name, n_gpu_layers=gpu_flag, n_ctx=self.config[args.model_lib]["basic"]["n_ctx"])
        elif args.model_lib == 'openai':
            self.model = OpenAI()

            
        elif args.model_lib == 'huggingface':
            self.device = "cuda" if args.use_gpu else "cpu"
            dtype = model_dtype_map[args.model_name_simplified]
            
            if 'xgen1' in args.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained("/export/share/bpang/pytorch_ckpts/exp_552_xgen3b_2624", ignore_mismatched_sizes=True, torch_dtype=dtype).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen2", trust_remote_code=True, torch_dtype=dtype)
            elif 'xgen2' in args.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained("/export/share/bpang/pytorch_ckpts/exp_901_xgen1b_5248", ignore_mismatched_sizes=True, torch_dtype=dtype).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen2", trust_remote_code=True, torch_dtype=dtype)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=dtype).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=dtype)
                
        else:
            raise NotImplementedError("Library not supported.")
        

    def predict(self, prompt, temperature_explicit=None, do_sample_explicit=None):
        
        kwargs = self.config[self.model_lib][self.task_nm]
        if temperature_explicit:
            kwargs["temperature"] = temperature_explicit
        if "do_sample" in kwargs and do_sample_explicit is not None:
            kwargs["do_sample"] = do_sample_explicit

        if self.model_lib == 'llama_cpp_python':
            try:
                # print(prompt)
                pred = self.model(prompt, **kwargs)['choices'][0]['text']
            except:
                pred = self.model(prompt[:4000], **kwargs)['choices'][0]['text']
        elif self.model_lib == 'openai':
            completion = self.model.chat.completions.create(
              model="gpt-4",
              messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
              ],
                max_tokens=kwargs["max_tokens"],
                temperature=kwargs["temperature"]
            )
            pred = completion.choices[0].message.content
        elif self.model_lib == 'huggingface':
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(input_ids, **kwargs)
            pred = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return pred
        


            
        


            