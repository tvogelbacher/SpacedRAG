import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = f"{test_params['interpreter_path']} -u spaced_gen_adv.py \
        --retrieval_emb_model {test_params['retrieval_emb_model']}\
        --score_function {test_params['score_function']}\
        --knowledge_base {test_params['knowledge_base']}\
        --split {test_params['split']}\
        --orig_beir_results {test_params['orig_beir_results']}\
        --knowledge_base_is_known {test_params['knowledge_base_is_known']}\
        --model_name {test_params['model_name']}\
        --model_config_path {test_params['model_config_path']}\
        --gpu_id {test_params['gpu_id']}\
        --adv_per_query {test_params['adv_per_query']}\
        --defense_emb_model {test_params['defense_emb_model']}\
        --query_versions {test_params['query_versions']}\
        --corpus_versions {test_params['corpus_versions']}\
        --corpus_length {test_params['corpus_length']}\
        --percentage {test_params['percentage']}\
        --optimization_steps {test_params['optimization_steps']}\
        --data_num {test_params['data_num']}\
        --seed {test_params['seed']}\
        --save_path {test_params['save_path']}\
        --name {log_name}\
        --verbosity {test_params['verbosity']}\
        > {log_file} &"
        
    os.system(cmd)

def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"logs/spaced_adv_gen_logs", exist_ok=True)
    log_name = f"{test_params['knowledge_base']}-{test_params['knowledge_base_is_known']}-{test_params['retrieval_emb_model']}-{test_params['model_name']}-adv-{test_params['adv_per_query']}-{test_params['query_versions']}-{test_params['corpus_versions']}"
    
    return f"logs/spaced_adv_gen_logs/{log_name}.txt", log_name


test_params = {
    # General
    'interpreter_path': '"C:/Users/test/Desktop/SpacedRAG/.venv/Scripts/python.exe"',
    
    # Retriever and knowledge base
    'retrieval_emb_model': "contriever",
    'score_function': 'cos_sim',
    'knowledge_base': "hotpotqa",
    'split': "test",
    'orig_beir_results': None,
    'knowledge_base_is_known': "True",

    # LLM setting
    'model_name': 'gpt-4.1-mini', 
    'model_config_path': 'model_configs/gpt4.1mini_config.json',
    'gpu_id': 0,

    # Attack
    'defense_emb_model': "simcse",
    'adv_per_query': 5,
    'query_versions': 25,
    'corpus_versions': 25,
    'corpus_length': 30,
    'percentage': 0.0000001, #0.000000015, #0.002,
    'optimization_steps': 3,
    'data_num': 1,
    'seed': 11, #73 nq, 78 msmarco, 94 hotpotqa
    'save_path': 'results/adv_spaced_targeted_results',

    # Log
    'verbosity': 2
}

run(test_params)