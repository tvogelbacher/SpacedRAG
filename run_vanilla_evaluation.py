import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = f"{test_params['interpreter_path']} -u vanilla_evaluation.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --model_config_path {test_params['model_config_path']}\
        --top_k {test_params['top_k']}\
        --use_truth {test_params['use_truth']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --name {log_name}\
        > {log_file} &"
        
    os.system(cmd)


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params['note'] != None:
        log_name = test_params['note']
    
    return f"logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name



test_params = {
    # venv interpreter
    'interpreter_path': '"C:/Users/test/Desktop/SpacedRAG/.venv/Scripts/python.exe"',
    # beir_info
    'eval_model_code': "ance",
    'eval_dataset': "nq",
    'split': "test",
    'query_results_dir': 'vanilla_eval',

    # LLM setting
    'model_name': 'gpt-4.1-mini',#'llama3.1_8b',
    'model_config_path': 'model_configs/gpt4.1mini_config.json',#'model_configs/llama3.1_8b_config.json', #'model_configs/gpt4.1mini_config.json',
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,

    # attack
    'attack_method': 'LM_targeted',
    'adv_per_query': 5,
    'score_function': 'cos_sim',
    'repeat_times': 10,
    'M': 10,
    'seed': 12,

    'note': None
}

for dataset in ['msmarco']:
    test_params['eval_dataset'] = dataset
    run(test_params)