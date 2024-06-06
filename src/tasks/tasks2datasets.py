task2dataset = {
    'summarization': ['cnn_dailymail', 'edinburghNLP_xsum'], 
    'text2sql': ['sql_create_context'], 
    'question_answering': ['hotpot_qa', 'databricks_dolly_15k'],
    'alpaca_eval': ['alpaca_eval'], 
    'mt_bench': ['mt_bench'],
    'trust_and_safety': ['adv_instruction', 'bbq', 'do_not_answer', 'privacy_leakage', 'social_chemistry_101', 'truthful_qa'],
    'mmlu': ['mmlu'],
    'math': ['gsm8k']
}

dataset2path = {
    'cnn_dailymail': './data/cnn_dailymail.csv',
    'edinburghNLP_xsum':'./data/edinburghNLP_xsum.csv',
    'sql_create_context':'./data/sql_create_context.csv',
    'hotpot_qa':'./data/hotpot_qa.csv',
    'databricks_dolly_15k':'./data/databricks_dolly_15k.csv',
    'alpaca_eval':'',
    'mt_bench': './data/mt_bench_question.jsonl',  
    'adv_instruction':'./data/adv_instruction.csv', 
    'bbq':'./data/bbq.csv', 
    'do_not_answer':'./data/do_not_answer.csv', 
    'privacy_leakage':'./data/privacy_leakage.csv', 
    'social_chemistry_101':'./data/social_chemistry_101.csv',
    'truthful_qa':'./data/truthful_qa.csv',
    'mmlu': './data/mmlu.csv',
    'gsm8k': './data/gsm8k.csv',
}