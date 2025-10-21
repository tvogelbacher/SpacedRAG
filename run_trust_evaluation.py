import os
import re

def run(test_params):
    log_name = get_log_name(test_params)
    if log_name + '.log' in os.listdir('logs'):
        print(f"log {log_name}.log already exists")
        # load the log file
        with open(f'logs/{log_name}.log', 'r', encoding="utf-8") as f:
            lines = f.readlines()
        # search if "Incorrect Answer Percentage:" in the log file with re
        if len(lines) > 0:
            incorrect_answer_percentage = re.search(r'Incorrect Answer Percentage:', lines[-1])
            if not incorrect_answer_percentage:
                print(f"log {log_name}.log is not complete, remove it")
                os.remove(f'logs/{log_name}.log')
            else:
                print(f"log {log_name}.log is complete")
                return
        else:
            print(f"log {log_name}.log is not complete, remove it")
            os.remove(f'logs/{log_name}.log')

    cmd = f"{test_params['interpreter_path']} -u trust_evaluation.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --model_config_path {test_params['model_config_path']}\
        --top_k {test_params['top_k']}\
        --defense_model {test_params['defense_model']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_type {test_params['attack_type']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --log_name {log_name} \
        --defend_method {test_params['defend_method']}\
        --removal_method {test_params['removal_method']}"    
    os.system(cmd)

def get_log_name(test_params):
    log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}"
    if test_params['removal_method'] != None:
        log_name += f"-removal-{test_params['removal_method']}"
    if test_params['defend_method'] != None:
        log_name += f"-defend-{test_params['defend_method']}"
    log_name += f"-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"
    if test_params['note'] != None:
        log_name = test_params['note']
    return log_name



test_params = {
    # venv interpreter
    'interpreter_path': '"C:/Users/test/Desktop/SpacedRAG/.venv/Scripts/python.exe"',
    
    'eval_model_code': "ance",
    'eval_dataset': "msmarco", # ['nq','hotpotqa', 'msmarco']
    'split': "test",
    'query_results_dir': 'trust_eval',
    'model_name': 'llama', #'gpt-4.1-mini',
    'model_config_path': 'model_configs/llama3.1_8b_config.json', #'model_configs/gpt4.1mini_config.json',  
    'top_k': 5,
    'defense_model': 'bert', # [simcse, bert]
    'gpu_id': 0,
    'attack_type': 'SpacedRAG', # ['SpacedRAG', 'PoisonedRAG']
    'attack_method': 'LM_targeted', # ['none', 'LM_targeted', 'hotflip', 'pia']
    'defend_method': 'none', # ['none', 'conflict', 'astute', 'instruct']
    'removal_method': 'kmeans', # ['kmeans', 'kmeans_ngram', 'none']
    'adv_per_query': 5, # poison rate = adv_per_query / top_k
    'score_function': 'cos_sim',
    'repeat_times': 10,
    'M': 10, # number of queries
    'seed': 12,
    'note': None
}

run(test_params)