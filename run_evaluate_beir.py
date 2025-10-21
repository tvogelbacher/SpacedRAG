import os

def run(test_params):

    result_output_path = get_result_output_path(test_params)

    cmd = f"{test_params['interpreter_path']} -u evaluate_beir.py \
        --model_code {test_params['model_code']}\
        --score_function {test_params['score_function']}\
        --top_k {test_params['top_k']}\
        --dataset {test_params['dataset']}\
        --split {test_params['split']}\
        --result_output {result_output_path}\
        --gpu_id {test_params['gpu_id']}\
        --per_gpu_batch_size {test_params['per_gpu_batch_size']}\
        --corpus_chunk_size {test_params['corpus_chunk_size']}\
        --max_length {test_params['max_length']}\
         &"
        
    os.system(cmd)

def get_result_output_path(test_params):
    # Generate a result output path
    os.makedirs("results/beir_results", exist_ok=True)
    result_output = f"results/beir_results/{test_params['dataset']}-{test_params['model_code']}-{test_params['score_function']}.json"
    
    return result_output

test_params = {
    # venv interpreter
    'interpreter_path': '"C:/Users/test/Desktop/SpacedRAG/.venv/Scripts/python.exe"',

    # beir_info
    'model_code': "ance",
    'score_function': 'cos_sim',
    'top_k': 20,
    'dataset': "msmarco",
    'split': "test",
    'gpu_id': 0,
    'per_gpu_batch_size': 64,
    'corpus_chunk_size': 10000,
    'max_length': 128
}


run(test_params)