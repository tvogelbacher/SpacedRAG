import argparse
import os
import json
import numpy as np
import torch
import requests
import nltk
from nltk.tokenize import sent_tokenize
from src.utils import load_beir_datasets, load_models, save_json, setup_seeds
from src.models import create_model
from src.prompts import wrap_prompt
from src.spaced_utils import *
from src.models.EmbeddingModel import EmbeddingModel
import time
from datetime import datetime
from itertools import combinations
from math import comb
import random

def query_llama(input, model_name, return_json: bool):
    url = 'https://router.huggingface.co/v1/chat/completions'
    access_token = os.environ.get('HF_ACCESS_TOKEN')
    assert access_token, "Please set the HF_ACCESS_TOKEN environment variable with your Hugging Face access token."
    headers = {
        'Authorization': f"Bearer {access_token}",
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_name,
        'temperature': 1,
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                     {'role': 'user', 'content': input}]
    }
    if return_json:
        data['response_format'] = {"type": "json_object"}
    
    retries = 0
    while retries < 5:
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError: {e}. Retrying...")
            time.sleep(20)
            retries += 1
    result = {'usage': response.json()['usage'], 'output': response.json()['choices'][0]['message']['content']}
    return result['output']

def query_gpt(input, model_name, return_json: bool):
    url = 'https://api.openai.com/v1/chat/completions'
    api_key = os.environ.get('OPENAI_API_KEY')
    assert api_key, "Please set the OPENAI_API_KEY environment variable with your OpenAI API key."
    headers = {
        'Authorization': f"Bearer {api_key}",
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_name,
        'temperature': 1,
        'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                     {'role': 'user', 'content': input}]
    }
    if return_json:
        data['response_format'] = {"type": "json_object"}

    retries = 0
    while retries < 5:
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError: {e}. Retrying...")
            time.sleep(10)
            retries += 1
    result = {'usage': response.json()['usage'], 'output': response.json()['choices'][0]['message']['content']}
    return result['output']

def parse_args():
    parser = argparse.ArgumentParser(description="test")

    # Retriever and knowledge base
    parser.add_argument(
        "--retrieval_emb_model",
        type=str,
        default="contriever",
        choices=["contriever-msmarco", "contriever", "ance"],
        help="By the attacker assumed embedding model used for retrieval"
    )
    parser.add_argument("--score_function", type=str, default='cos_sim', choices=['dot', 'cos_sim'], help="Similarity function used only for retrieval")
    parser.add_argument("--knowledge_base", type=str, default="nq", help="Dataset used as knowledge base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of retrieval model on the original beir dataset')
    parser.add_argument("--knowledge_base_is_known", type=str, default="True", help="True for whitebox, False for blackbox with context retrieval attack")
    
    # LLM setting
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument('--gpu_id', type=int, default=0)

    # Attack
    parser.add_argument(
        "--defense_emb_model",
        type=str,
        default="simcse",
        choices=["simcse", "bert"],
        help="By the attacker assumed embedding model used for TrustRag k-means defense"
    )
    parser.add_argument("--adv_per_query", type=int, default=5, help="Number of adv_text per query")
    parser.add_argument("--query_versions", type=int, default=5, help="Number of query versions to generate per query")
    parser.add_argument("--corpus_versions", type=int, default=5, help="Number of corpus versions to generate per query")
    parser.add_argument("--corpus_length", type=int, default=30, help="Rough length of each corpus to generate")
    parser.add_argument("--percentage", type=float, help="Percentage of sets of adv_per_query adv texts to consider for similarity computation")
    parser.add_argument("--optimization_steps", type=int, default=1, help="Number of optimization steps to perform (1 step goes through all adv texts once)")
    parser.add_argument("--data_num", type=int, default=100, help="Number of query samples to generate adv_text for")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility of chosen queries")
    parser.add_argument("--save_path", type=str, default="results/adv_spaced_targeted_results", help="Save path of adv texts.")    

    # Log
    parser.add_argument("--verbosity", type=int, default=2, help="Log verbosity level")
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    args = parser.parse_args()
    print(f"{args}\n")
    return args

def get_radius(results, query_id, number_of_included_clean, knowledge_base_is_known):
    """Identify radius r for the hypersphere. Incude clean texts into hypershere if former attempts with smaller r failed."""
    if knowledge_base_is_known == "True":
        included_clean_texts = []
        r = 0
        for i in range(number_of_included_clean+1):
            top_counter_id = list(results[query_id].keys())[i]
            top_counter_score = results[query_id][top_counter_id]
            r = top_counter_score
            included_clean_texts.append(top_counter_id)
        included_clean_texts = included_clean_texts[:-1] # remove text that marks the radius
        return r, included_clean_texts
    
    elif knowledge_base_is_known == "False":
        raise NotImplementedError
    
def get_correct_answer(args, llm, question, query_id, corpus, qrels):
    """Get the correct answer for the current query"""
    """if args.knowledge_base == "hotpotqa":
        # Directly load correct answer. Only hotpotqa.
        query_location = f"datasets/{args.knowledge_base}/queries.jsonl"
        with open(query_location, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry['_id'] == query_id:
                    correct_answer = entry['metadata']['answer']
    else:"""
    # Generate correct answer using ground truth contexts
    qrel_ids = list(qrels[query_id].keys())
    ground_truths = [corpus[qrel_id]['text'] for qrel_id in qrel_ids]
    ground_truth_prompt = wrap_prompt(question, ground_truths, 4)
    response_q = llm.query(question)
    response_w_gt = llm.query(ground_truth_prompt)

    print(f"response_q: {response_q.encode('utf8')}")
    print(f"response_w_gt: {response_w_gt.encode('utf8')}")
    # Keep the shorter correct answer
    if response_q.lower() in response_w_gt.lower():
        correct_answer = response_q
    elif response_w_gt.lower() in response_q.lower():
        correct_answer = response_w_gt
    else:
        correct_answer = ""
        
    return correct_answer
    
def get_baseline(args, llm, corpus, question, correct_answer, query_emb, r, included_clean_texts, retrieval_model, defense_model, scale_boundaries, relevant_data):
    max_retries = 5 # Number of retries for LLM calls in case of json keyerrors

    # Generate adversarial subtexts S for retrieval condition
    gen_adv_prompt = build_adv_query_prompt(question, args.query_versions)
    for attempt in range(1, max_retries + 1):
        try:
            response = query_gpt(gen_adv_prompt, model_name='gpt-4.1-mini', return_json=True)
            adv_queries = json.loads(response)

            adv_subtexts_S = extract_llm_response(adv_queries, "query", args.query_versions)
            adv_subtexts_S = {f"S{k+1}": adv_subtexts_S[f"query{k+1}"] for k in range(args.query_versions)}
            break
        except KeyError as e:
            print(f"KeyError: {e}. Retrying... ({attempt}/{max_retries})")
            if attempt == max_retries:
                print("Max retries reached. Skipping this query.")
                continue

    if args.verbosity >= 2:
        print("Generated queries:")
        for key, value in adv_subtexts_S.items():
            print(f"{key}: {value}")
    
    # To test just subtext I as adv text
    #adv_subtexts_S = {f"S{k+1}": "" for k in range(args.query_versions)}

    # Generate incorrect answer and adversarial subtext I for generation condition
    gen_adv_prompt = build_adv_corpus_prompt(question, correct_answer, args.corpus_versions, args.corpus_length)
    for attempt in range(1, max_retries + 1):
        try:
            response = query_gpt(gen_adv_prompt, model_name='gpt-4.1-mini', return_json=True)
            adv_corpus = json.loads(response)
            incorrect_answer = adv_corpus["incorrect_answer"]

            adv_subtexts_I = extract_llm_response(adv_corpus, "corpus", args.corpus_versions)
            adv_subtexts_I = {f"I{k+1}": adv_subtexts_I[f"corpus{k+1}"] for k in range(args.corpus_versions)}
            break
        except KeyError as e:
            print(f"KeyError: {e}. Retrying... ({attempt}/{max_retries})")
            if attempt == max_retries:
                print("Max retries reached. Skipping this query.")
                continue

    print(f"\nGenerated incorrect answer: {incorrect_answer}")
    if args.verbosity >= 2:
        print("Generated corpora:")
        for key, value in adv_subtexts_I.items():
            print(f"{key}: {value}")

    # Adjust number of adv texts to generate if clean texts are accepted to be in hypersphere
    adjusted_adv_per_query = args.adv_per_query - len(included_clean_texts)

    # Check misleading capabilities of the generated subtexts I
    adv_subtexts_I_temp = adv_subtexts_I.copy()
    for idx, subtext in adv_subtexts_I.items():
        gen_check_prompt = wrap_prompt(question, [subtext], 4)
        response = llm.query(gen_check_prompt)
        if incorrect_answer.lower() not in response.lower():
            adv_subtexts_I_temp.pop(idx)
    print(f"\nNumber of generated corpora leading to the incorrect answer: {len(adv_subtexts_I_temp)} out of {args.corpus_versions}")
    
    if len(adv_subtexts_I_temp) < adjusted_adv_per_query:
        print(f"Only {len(adv_subtexts_I_temp)} out of {args.corpus_versions} generated corpora lead to the incorrect answer. Continuing with all generated I.\n")
    else:
        adv_subtexts_I = adv_subtexts_I_temp
    
    # Calculate embeddings of all possible adv text SI
    retrieval_adv_embs = compute_adv_embeddings(adv_subtexts_S, adv_subtexts_I, retrieval_model)
    print(f"\nNumber of possible adv texts: {len(retrieval_adv_embs)}")
    defense_adv_embs = compute_adv_embeddings(adv_subtexts_S, adv_subtexts_I, defense_model)

    # Compute embeddings of included clean texts
    defense_clean_embs = compute_clean_embeddings(included_clean_texts.keys(), corpus, defense_model)

    # Identify possible adv text SI outside of hypersphere
    sim_adv_texts_to_query = compute_sim_to_query(retrieval_adv_embs, query_emb, args.score_function)
    adv_SI_outside_r = get_adv_texts_outside_r(retrieval_adv_embs, sim_adv_texts_to_query, r)
    relevant_data["texts_outside_r"].append(len(adv_SI_outside_r))

    # Log relevant data
    maxsq = max(sim_adv_texts_to_query.values())
    relevant_data["max_query_sim"].append(maxsq)
    avgsq = sum(sim_adv_texts_to_query.values()) / len(sim_adv_texts_to_query)
    relevant_data["avg_query_sim"].append(avgsq) 
    minsq = min(sim_adv_texts_to_query.values())
    relevant_data["min_query_sim"].append(minsq)
    if args.verbosity >= 2:
        print(f"Max sim to query: {maxsq}")
        print(f"Avg sim to query: {avgsq}")
        print(f"Min sim to query: {minsq}\n")

    # Compute all subsets of size 2 of SI (i.e. two adv texts) and cache their similarity
    adv_2_sets = build_adv_sets(list(adv_subtexts_S.keys()), list(adv_subtexts_I.keys()), 2, 1, args.verbosity)
    adv_2_sets = filter_adv_sets(adv_2_sets, adv_SI_outside_r)

    # Also consider similarity of included clean texts
    adv_clean_2_sets = set((adv_text, clean_text) for adv_text in defense_adv_embs.keys() for clean_text in included_clean_texts.keys())
    if len(included_clean_texts) > 1:
        clean_2_sets = set((clean1, clean2) for i, clean1 in enumerate(included_clean_texts.keys()) for clean2 in list(included_clean_texts.keys())[i+1:])
        adv_clean_2_sets = adv_clean_2_sets.union(clean_2_sets)
    all_embs = {**defense_adv_embs, **defense_clean_embs}
    all_2_sets = adv_clean_2_sets.union(adv_2_sets)
    all_2_sets_sim = compute_2_set_sim(all_2_sets, all_embs)
    
    # Calculate total number of possible sets with original adv_per_query
    S_combos = comb(args.query_versions, args.adv_per_query)
    I_combos = comb(args.corpus_versions, args.adv_per_query)
    total_per_SI = factorial(args.adv_per_query)
    original_total_sets = S_combos * I_combos * total_per_SI

    # Calculate total number of possible sets with adjusted adv_per_query
    S_combos = comb(args.query_versions, adjusted_adv_per_query)
    I_combos = comb(len(adv_subtexts_I), adjusted_adv_per_query)
    total_per_SI = factorial(adjusted_adv_per_query)
    adjusted_total_sets = S_combos * I_combos * total_per_SI

    # Adjust percentage to keep the same number of considered sets
    # If there are less possible sets than originally intended, consider all sets
    if adjusted_total_sets < (original_total_sets * args.percentage):
        percentage = 1
    else:
        percentage = original_total_sets / adjusted_total_sets * args.percentage

    # Compute all subsets of size adv_per_query - clean texts of S and I, stored with substitutes (S_index, I_index)
    adv_sets = build_adv_sets(list(adv_subtexts_S.keys()), list(adv_subtexts_I.keys()), args.adv_per_query-len(included_clean_texts), percentage, args.verbosity)
    if args.verbosity >= 2:
        print(f"Number of possible subset combinations to be evaluated: {len(adv_sets)}")

    # Identify and filter subsets containing an adv text outside the hypersphere
    if args.verbosity >= 2:
        print(f"Number of adv texts outside hypersphere: {len(adv_SI_outside_r)}")
    if args.verbosity >= 3:
        print("Adv texts outside hypersphere:")
        for key, value in adv_SI_outside_r.items(): 
            print(f"{key}: {value}")
    adv_sets = filter_adv_sets(adv_sets, adv_SI_outside_r)
    if args.verbosity >= 2:
        print(f"Remaining Number of subset combinations with all adv texts in hypersphere: {len(adv_sets)}")
    
    # Special flag to indicate retry with bigger radius if no valid adv_sets remain
    if not adv_sets:
        return None, None, None 

    # Also consider included clean texts in the sets
    adv_sets_with_clean = [adv_set + tuple(included_clean_texts.keys()) for adv_set in adv_sets]
            
    sets_cluster_avg_sim = {}
    for adv_set in tqdm(adv_sets_with_clean, desc=f"{datetime.now().strftime('%H:%M:%S')}: Calculating baseline"):
        # K-means cluster the adv texts in the current subset into 2 clusters
        set_embs = {k: all_embs[k] for k in adv_set}
        cluster0, cluster1 = kmeans_cluster(list(set_embs.values()), list(set_embs.keys()))

        # Calculate avg similarity for the 2 kmeans clusters
        cluster0_avg_sim = compute_set_sim(cluster0, all_2_sets_sim)
        cluster1_avg_sim = compute_set_sim(cluster1, all_2_sets_sim)
        sets_cluster_avg_sim[adv_set] = (cluster0_avg_sim, cluster1_avg_sim)

        # Update avg sim scale boundaries if new min or max is found
        if cluster0_avg_sim > scale_boundaries["avg_sim_scale"][1]:
            scale_boundaries["avg_sim_scale"][1] = cluster0_avg_sim
        if cluster0_avg_sim < scale_boundaries["avg_sim_scale"][0] and cluster0_avg_sim != 0:
            scale_boundaries["avg_sim_scale"][0] = cluster0_avg_sim
        if cluster1_avg_sim > scale_boundaries["avg_sim_scale"][1]:
            scale_boundaries["avg_sim_scale"][1] = cluster1_avg_sim
        if cluster1_avg_sim < scale_boundaries["avg_sim_scale"][0] and cluster1_avg_sim != 0:
            scale_boundaries["avg_sim_scale"][0] = cluster1_avg_sim
    
    print(f"scale_boundaries: {scale_boundaries}")
    
    # Normalize scores and compute final score for every subset
    sets_weighted_sim = {}
    for adv_set, scores in sets_cluster_avg_sim.items():
        # Normalize avg sim to [0, 1] based on scale boundaries
        norm_cluster0_avg_sim = normalize(scores[0], scale_boundaries["avg_sim_scale"])
        norm_cluster1_avg_sim = normalize(scores[1], scale_boundaries["avg_sim_scale"])
        # Calculate final score with dynamic weights dependant on normalized values, setting them into proportion of their severity
        sets_weighted_sim[adv_set] = get_weighted_score(norm_cluster0_avg_sim, norm_cluster1_avg_sim)

    # Identify best subset and log relevant data
    adv_texts = min(sets_weighted_sim, key=sets_weighted_sim.get)
    min_final_score = sets_weighted_sim[adv_texts]
    if args.verbosity >= 3:
        print(f"Final score of best subset: {min_final_score}")
    baseline_cluster_avg_sim = sets_cluster_avg_sim[adv_texts]
    print(f"Clusterwise average similarity values for chosen baseline: {baseline_cluster_avg_sim}")

    relevant_data["baseline_cluster_avg_sim"].append(baseline_cluster_avg_sim)

    # Remove included clean texts from adv_texts for further optimization
    adv_texts = tuple(text for text in adv_texts if text not in included_clean_texts.keys())
    print(f"Chosen adv texts as baseline (without clean texts): {adv_texts}\n") #!!!

    # Translate the substitutes to text
    adv_texts = [f"{adv_subtexts_S[adv_text[0]]} {adv_subtexts_I[adv_text[1]]}" for adv_text in adv_texts]

    return adv_texts, baseline_cluster_avg_sim, incorrect_answer

def optimize_avg_sim(args, corpus, adv_texts, query_emb, r, included_clean_texts, baseline_score, retrieval_model, defense_model, scale_boundaries, relevant_data):
    """
    Optimize clusterwise average similarity of adversarial texts through paraphrasing on sentence level.
    For each adv text, paraphrase each sentence (one at a time), check if the new text is still inside the hypersphere
    and if the average pairwise similarity with the other adv texts decreases. If so, keep the paraphrased version and
    repeat for the current sentence; otherwise, move to the next one.
    """         
    optimized_texts = adv_texts.copy()
    best_score = 0
    best_cluster_sim = baseline_score
    for idx, adv_text in tqdm(enumerate(adv_texts), desc=f"{datetime.now().strftime('%H:%M:%S')}: Optimizing avg sim"):
        if args.verbosity >= 3:
            print(f"\n###Optimizing adv text {idx+1}/{len(adv_texts)}###")
        current_text = adv_text
        sentences = sent_tokenize(current_text)
        for i in range(len(sentences)):
            while True:
                if args.verbosity >= 3:
                    print(f"Optimizing sentence {i+1}/{len(sentences)}: {sentences[i]}")
                # Paraphrase the i-th sentence using GPT
                opt_prompt = build_sim_optimization_prompt(sentences[i], 10)

                response = query_gpt(opt_prompt, model_name='gpt-4.1-mini', return_json=True)
                paraphrases = json.loads(response)
                paraphrases = extract_llm_response(paraphrases, 'sentence', 10)

                best_text = current_text
                found_better = False

                for para in paraphrases.values():
                    # Build new text with paraphrased sentence
                    new_sentences = sentences.copy()
                    new_sentences[i] = para
                    new_text = ' '.join(new_sentences)

                    # Compute embedding and similarity to query
                    new_emb = retrieval_model.embed(new_text)
                    sim_to_query = get_sim(args.score_function, new_emb, query_emb)
                    if sim_to_query < r:
                        continue  # Not in hypersphere

                    # Compute defense embeddings, also consider included clean texts
                    other_texts = optimized_texts.copy() + list(included_clean_texts.values())
                    other_texts[idx] = new_text
                    all_embs = [defense_model.embed(t) for t in other_texts]
                    all_embs = {i: all_embs[i] for i in range(len(all_embs))}

                    cluster0, cluster1 = kmeans_cluster(list(all_embs.values()), list(all_embs.keys()))

                    # Generate all possible combinations of size 2 for cluster0 and cluster1
                    cluster0_pairs = list(combinations(cluster0, 2))
                    cluster1_pairs = list(combinations(cluster1, 2))
                    cluster0_pairs_sim = compute_2_set_sim(cluster0_pairs, all_embs)
                    cluster1_pairs_sim = compute_2_set_sim(cluster1_pairs, all_embs)

                    cluster0_avg_sim = compute_set_sim(cluster0, cluster0_pairs_sim)
                    cluster1_avg_sim = compute_set_sim(cluster1, cluster1_pairs_sim)

                    # Update avg sim scale boundaries if new min or max is found
                    if cluster0_avg_sim > scale_boundaries["avg_sim_scale"][1]:
                        scale_boundaries["avg_sim_scale"][1] = cluster0_avg_sim
                    elif cluster0_avg_sim < scale_boundaries["avg_sim_scale"][0] and cluster0_avg_sim != 0:
                        scale_boundaries["avg_sim_scale"][0] = cluster0_avg_sim
                    if cluster1_avg_sim > scale_boundaries["avg_sim_scale"][1]:
                        scale_boundaries["avg_sim_scale"][1] = cluster1_avg_sim
                    elif cluster1_avg_sim < scale_boundaries["avg_sim_scale"][0] and cluster1_avg_sim != 0:
                        scale_boundaries["avg_sim_scale"][0] = cluster1_avg_sim

                    # Normalize scores to [0, 1] based on scale boundaries
                    norm_cluster0_avg_sim = normalize(cluster0_avg_sim, scale_boundaries["avg_sim_scale"])
                    norm_cluster1_avg_sim = normalize(cluster1_avg_sim, scale_boundaries["avg_sim_scale"])
                    avg_score = get_weighted_score(norm_cluster0_avg_sim, norm_cluster1_avg_sim)

                    # Renormalize best score on new scale boundaries to compare with
                    norm_best_cluster0_sim = normalize(best_cluster_sim[0], scale_boundaries["avg_sim_scale"])
                    norm_best_cluster1_sim = normalize(best_cluster_sim[1], scale_boundaries["avg_sim_scale"])
                    best_score = get_weighted_score(norm_best_cluster0_sim, norm_best_cluster1_sim)

                    # Compare with current version
                    if avg_score < best_score:
                        best_score = avg_score
                        best_text = new_text
                        best_cluster_sim = (cluster0_avg_sim, cluster1_avg_sim)
                        found_better = True

                if found_better and best_text != current_text:
                    if args.verbosity >= 3:
                        print(f"Found better paraphrase with avg_score: {best_score}")
                        print(f"Intra Cluster Sim: {best_cluster_sim}")
                    # If any paraphrase is better, update and repeat for this sentence
                    current_text = best_text
                    sentences = sent_tokenize(current_text)
                else:
                    # No improvement, move to next sentence
                    break
        optimized_texts[idx] = current_text
    print(f"best_cluster_sim after optimization: {best_cluster_sim}\n")
    relevant_data["opt_cluster_avg_sim"].append(best_cluster_sim)

    return optimized_texts, best_cluster_sim

def optimize_max_rouge(adv_texts, question, correct_answer, incorrect_answer, threshold, query_emb, r, included_clean_texts, retrieval_model, defense_model, relevant_data):
    """
    For every adv text pair until all are below the threshold, identify the longest common sequence.
    Pick a random lcs word that is not in the question, correct or incorrect answer. 
    Identify the sentence it's in and the instance number for it.
    Get a number of paraphrases for this sentence with the lcs word replaced.
    Pick the one that minimizes the maximum rouge-L with all other adv texts in the cluster.
    """
    # Identify words to keep during rouge optimization
    words_to_keep = [sanitize_text(w) for w in extract_names(question)]
    words_to_keep += remove_stopwords(sanitize_text(incorrect_answer)).split()
    words_to_keep += remove_stopwords(sanitize_text(correct_answer)).split()
    print(f"Words to keep during rouge optimization: {words_to_keep}")
    max_retries = 10
    cluster_texts = adv_texts.copy() + list(included_clean_texts.values())
    retries = 0
    while True:
        pairs = list(combinations(range(len(cluster_texts)), 2))
        max_rouge = 0

        # Identify the pair with the highest rouge-L
        pair = None
        adv_cross_rouge_values = []
        clean_rouge_values = []
        adv_rouge_values = []
        cross_rouge_values = []
        for t1_idx, t2_idx in pairs:
            t1 = cluster_texts[t1_idx]
            t2 = cluster_texts[t2_idx]

            rouge = calculate_max_rougeL([t2], t1)

            if t1 not in included_clean_texts.values() and t2 not in included_clean_texts.values():
                adv_rouge_values.append(rouge)
            elif t1 in included_clean_texts.values() and t2 in included_clean_texts.values():
                clean_rouge_values.append(rouge)
                continue # Don't consider clean-clean pairs for max rouge
            else:
                cross_rouge_values.append(rouge)

            adv_cross_rouge_values.append(rouge)
            if rouge > max_rouge:
                max_rouge = rouge
                pair = (t1_idx, t2_idx)

        # If all pairs are below threshold, stop
        if max_rouge < threshold:
            print(f"max rouge {max_rouge}")
            relevant_data["adv_cross_rouge_values"].append(adv_cross_rouge_values)
            relevant_data["clean_rouge_values"].append(clean_rouge_values)
            relevant_data["adv_rouge_values"].append(adv_rouge_values)
            relevant_data["cross_rouge_values"].append(cross_rouge_values)
            break
        
        t1_idx, t2_idx = pair
        t1 = cluster_texts[t1_idx]
        t2 = cluster_texts[t2_idx]
        t1_sanitized = sanitize_text(t1)
        t2_sanitized = sanitize_text(t2)

        # Identify the longest common sequence
        lcs1 = calculate_lcs(t1_sanitized, t2_sanitized)
        # If 2 different lcs exist
        lcs2 = calculate_lcs(t2_sanitized, t1_sanitized)
        lcs = lcs1 + [w for w in lcs2 if w not in lcs1]
        lcs_words = [w for w in lcs if w not in words_to_keep]
        if not lcs_words:
            print("Stopping rouge optimization as filtered LCS is empty. Retrying with bigger radius")
            return None

        # Pick a random lcs word and choose one of the texts to replace it in
        lcs_word = random.choice(lcs_words)
        print(f"Chosen LCS word: {lcs_word}")
        # Only pick randomly if both are not clean, otherwise pick the adversarial one
        if t1 in included_clean_texts.values() and t2 not in included_clean_texts.values():
            chosen_text_id = t2_idx
        elif t2 in included_clean_texts.values() and t1 not in included_clean_texts.values():
            chosen_text_id = t1_idx
        else:
            chosen_text_id = random.choice([t1_idx, t2_idx])
        chosen_text = cluster_texts[chosen_text_id]
        sentences = sent_tokenize(chosen_text)

        # Identify the sentence with the lcs word and its instance number
        #!!!
        instance_number = 0
        sentence_idx = None
        for idx, sent in enumerate(sentences):
            sent = sanitize_text(sent)
            count = sent.split().count(lcs_word)
            if count > 0:
                instance_number += 1
                sentence_idx = idx
                break

        opt_prompt = build_rouge_optimization_prompt(sentences[sentence_idx], lcs_word, instance_number, 5)
        response = query_gpt(opt_prompt, model_name='gpt-4.1-mini', return_json=True)
        new_versions = json.loads(response)
        new_versions = extract_llm_response(new_versions, 'sentence', 5)
        best_version = sentences[sentence_idx]
        best_max_rouge = max_rouge
        found_better = False
        for version in new_versions.values():
            candidate_sentences = sentences.copy()
            candidate_sentences[sentence_idx] = version
            candidate_text = ' '.join(candidate_sentences)

            # Compute embedding and similarity to query
            candidate_emb = retrieval_model.embed(candidate_text)
            sim_to_query = get_sim(args.score_function, candidate_emb, query_emb)
            if sim_to_query < r:
                continue  # Not in hypersphere

            candidate_max_rouge = calculate_max_rougeL([cluster_texts[other_idx] for other_idx in range(len(cluster_texts)) if other_idx != chosen_text_id], candidate_text)
            print(f"Candidate max rouge: {candidate_max_rouge}") #!!!
            if candidate_max_rouge < best_max_rouge:
                best_max_rouge = candidate_max_rouge
                best_version = version
                found_better = True
            # If equal, keep a new version as lcs can stay the same e.g. through another instance of the replaced word occurring again
            elif not found_better and candidate_max_rouge == best_max_rouge:
                best_version = version

        if not found_better:
            retries += 1
            print(f"No better paraphrase found. Retry {retries}/{max_retries}.")
        else:
            retries = 0

        if best_version != sentences[sentence_idx]:
            sentences[sentence_idx] = best_version
            cluster_texts[chosen_text_id] = ' '.join(sentences)
        # Special flag to indicate retry with bigger radius if max rouge cannot be reduced anymore
        if retries == max_retries:
            print("Max retries reached without improvement, stopping rouge optimization.")
            return None
        
    # Recalculate avg sim of final clusters for logging
    all_embs = {t: defense_model.embed(t) for t in cluster_texts}
    cluster0, cluster1 = kmeans_cluster(list(all_embs.values()), list(all_embs.keys()))
    cluster0_pairs = list(combinations(cluster0, 2))
    cluster1_pairs = list(combinations(cluster1, 2))
    cluster0_pairs_sim = compute_2_set_sim(cluster0_pairs, all_embs)
    cluster1_pairs_sim = compute_2_set_sim(cluster1_pairs, all_embs)

    avg_sim_values = []
    clean_sim_values = []
    cross_sim_values = []
    # Sort sim values into the correct lists
    for pair, sim in list(cluster0_pairs_sim.items()) + list(cluster1_pairs_sim.items()):
        t1, t2 = pair
        is_clean1 = t1 in included_clean_texts.values()
        is_clean2 = t2 in included_clean_texts.values()
        if not is_clean1 and not is_clean2:
            avg_sim_values.append(sim)
        elif is_clean1 and is_clean2:
            clean_sim_values.append(sim)
        else:
            cross_sim_values.append(sim)
    
    relevant_data["adv_sim_values"].append(avg_sim_values)
    relevant_data["clean_sim_values"].append(clean_sim_values)
    relevant_data["cross_sim_values"].append(cross_sim_values)

    # Remove included clean texts from cluster_texts for further processing
    cluster_texts = [text for text in cluster_texts if text not in included_clean_texts.values()]

    print(f"New clusters after rouge optimization: {cluster_texts}\n")
    return cluster_texts

def spaced_rag(args):
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    # Load retrieval embedding models
    retrieval_model_, retrieval_c_model, retrieval_tokenizer, retrieval_get_emb = load_models(args.retrieval_emb_model)
    retrieval_model_.eval()
    retrieval_model_.to(device)
    retrieval_c_model.eval()
    retrieval_c_model.to(device)
    retrieval_model = EmbeddingModel(
        identity=args.retrieval_emb_model,
        tokenizer=retrieval_tokenizer,
        model=retrieval_model_,
        c_model=retrieval_c_model,
        get_emb=retrieval_get_emb
    )

    # Load defense embedding models
    defense_model_, defense_c_model, defense_tokenizer, defense_get_emb = load_models(args.defense_emb_model)
    defense_model_.eval()
    defense_model_.to(device)
    defense_c_model.eval()
    defense_c_model.to(device)
    defense_model = EmbeddingModel(
        identity=args.defense_emb_model,
        tokenizer=defense_tokenizer,
        model=defense_model_,
        c_model=defense_c_model,
        get_emb=defense_get_emb
    )

    # Load LLM
    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
    llm = create_model(args.model_config_path)

    # Load dataset used as knowledge base
    corpus, queries, qrels = load_beir_datasets(args.knowledge_base, args.split)
    query_ids = list(queries.keys())

    # Load BEIR similarity results
    if args.orig_beir_results in [None, 'None']:
        print(f"Please evaluate on BEIR first -- {args.retrieval_emb_model} on {args.knowledge_base}")
        print("Now try to get beir eval results from results/beir_results/...")
        args.orig_beir_results = f"results/beir_results/{args.knowledge_base}-{args.retrieval_emb_model}-{args.score_function}.json"
    with open(args.orig_beir_results, 'r') as f:
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Get beir_results from {args.orig_beir_results}. \n")
        results = json.load(f)
    
    # Shuffle queries for random order
    setup_seeds(args.seed)
    np.random.shuffle(query_ids)

    # Gather relevant data over all queries
    relevant_data = {
        "max_query_sim": [],
        "avg_query_sim": [],
        "min_query_sim": [],
        "baseline_cluster_avg_sim": [],
        "opt_cluster_avg_sim": [],
        "adv_sim_values": [],
        "clean_sim_values": [],
        "cross_sim_values": [],
        "adv_cross_rouge_values": [],
        "adv_rouge_values": [],
        "clean_rouge_values": [],
        "cross_rouge_values": [],
        "texts_outside_r": [],
        "avg_sim_hypersphere_edge": []
    }
    adv_targeted_results = {}

    processed_queries = set()
    counter = 0
    try:
        while len(adv_targeted_results) < args.data_num:
            query_id = query_ids[counter]
            counter += 1
            processed_queries.add(query_id)

            print(f"########## Processing query: {len(adv_targeted_results)+1}/{args.data_num} ##########")
            question = queries[query_id]
            print(f"Query: {question}")

            # Calculate question embedding
            query_emb = retrieval_model.embed(question)

            # Get correct answer for the current query
            correct_answer = get_correct_answer(args, llm, question, query_id, corpus, qrels)

            if not correct_answer:
                if args.verbosity >= 3:
                    print(f"Skipping query {query_id} as no correct answer evaluable with substring matching could be generated.")
                continue
            print(f"\nCorrect answer: {correct_answer}")

            # Retry loop in case no valid adv_sets remain after filtering or max rouge-L is too high
            number_of_included_clean = 0
            while number_of_included_clean < args.adv_per_query:
                # Identify radius r of hypershere in white or black-box setting
                r, included_clean_texts = get_radius(results, query_id, number_of_included_clean, args.knowledge_base_is_known)
                print(f"\nRadius of hypershere: {r}")

                # get included clean texts by their ids
                included_clean_texts = {text_id: corpus[text_id]['text'] for text_id in included_clean_texts}

                # Keep track of avg sim boundaries per query for normalization
                scale_boundaries = {
                    "avg_sim_scale": [1, 0]
                }
                # Get baseline adv texts
                adv_texts, best_cluster_sim, incorrect_answer = get_baseline(args, llm, corpus, question, correct_answer, query_emb, r, included_clean_texts, retrieval_model, defense_model, scale_boundaries, relevant_data)
                if adv_texts is None:
                    print("Retrying query with bigger radius as no generated set has all its texts inside the hypersphere.")
                    number_of_included_clean += 1
                    continue

                # Optimize avg sim of adv texts through paraphrasing on sentence level
                for step in range(args.optimization_steps):
                    adv_texts, best_cluster_sim = optimize_avg_sim(args, corpus, adv_texts, query_emb, r, included_clean_texts, best_cluster_sim, retrieval_model, defense_model, scale_boundaries, relevant_data)
                
                # Optimize max rouge-L inside clusters
                adv_texts = optimize_max_rouge(adv_texts, question, correct_answer, incorrect_answer, 0.25, query_emb, r, included_clean_texts, retrieval_model, defense_model, relevant_data)
                
                if adv_texts is None:
                    print("Retrying query with bigger radius as the max rouge-L is still too high.")
                    number_of_included_clean += 1
                    continue

                break

            # Compute distance of adv texts to the edge of the hypersphere
            adv_embs = {i: retrieval_model.embed(adv_texts[i]) for i in range(len(adv_texts))}
            all_sim_to_query = compute_sim_to_query(adv_embs, query_emb, args.score_function)
            all_dist_to_r = {i: all_sim_to_query[i] - r for i in range(len(adv_texts))}
            avg_dist_to_r = sum(all_dist_to_r.values()) / len(all_dist_to_r)
            relevant_data["avg_sim_hypersphere_edge"].append(avg_dist_to_r)

            if args.verbosity >= 2:
                print("Average distance of adv texts to the edge of the hypershere:")
                print(f"{avg_dist_to_r}")
            if args.verbosity >= 3:
                print("Distance per adv texts:")
                for key, value in all_dist_to_r.items(): 
                    print(f"{key+1}: {value}") 

            # Fill up adv_texts with empty texts if less than adv_per_query texts were generated due to included clean texts or other fail conditions met
            while len(adv_texts) < args.adv_per_query:
                adv_texts.append("")   
        
            # Save the results
            adv_targeted_results[query_id] = {
                'id': query_id,
                'question': question,
                'correct answer': correct_answer,
                "incorrect answer": incorrect_answer,
                "adv_texts": [adv_texts[i] for i in range(args.adv_per_query)],
            }
            
            os.makedirs(args.save_path, exist_ok=True)
            save_json(adv_targeted_results, os.path.join(args.save_path, f'{args.knowledge_base}.json'))
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Avg sim of a cluster over all queries (baseline): {sum([s1 + s2 for s1, s2 in relevant_data['baseline_cluster_avg_sim']]) / (2 * len(relevant_data['baseline_cluster_avg_sim']))}")
    print(f"Avg sim of a cluster over all queries (optimized): {sum([s1 + s2 for s1, s2 in relevant_data['opt_cluster_avg_sim']]) / (2 * len(relevant_data['opt_cluster_avg_sim']))}")
    print(f"Avg distance of adv texts to the edge of the hypershere: {sum(relevant_data['avg_sim_hypersphere_edge']) / len(relevant_data['avg_sim_hypersphere_edge'])}")
    print(f"Avg ROUGE-L score of the adv texts to each other and potentially includede clean texts (optimized): {sum([sum(texts) for texts in relevant_data['adv_cross_rouge_values']]) / sum([len(texts) for texts in relevant_data['adv_cross_rouge_values']])}")
    #if args.verbosity >= 2:
    #    print(f"\nNumber of adv texts outside hypersphere averaged over all queries: {sum(relevant_data['texts_outside_r'])/len(relevant_data['texts_outside_r'])}")

    print(f"\nrelevant_data[adv_sim_values]: {relevant_data['adv_sim_values']}")
    print(f"\nrelevant_data[clean_sim_values]: {relevant_data['clean_sim_values']}")
    print(f"\nrelevant_data[cross_sim_values]: {relevant_data['cross_sim_values']}")
    print(f"\nrelevant_data[adv_cross_rouge_values]: {relevant_data['adv_cross_rouge_values']}")
    print(f"\nrelevant_data[clean_rouge_values]: {relevant_data['clean_rouge_values']}")
    print(f"\nrelevant_data[adv_rouge_values]: {relevant_data['adv_rouge_values']}")
    print(f"\nrelevant_data[cross_rouge_values]: {relevant_data['cross_rouge_values']}")

if __name__ == "__main__":
    args = parse_args()
    spaced_rag(args)