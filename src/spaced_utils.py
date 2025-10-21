import json
import torch
import random
from itertools import combinations, permutations
from math import factorial
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from rouge_score import rouge_scorer
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nlp = spacy.load("en_core_web_sm")

def clean_response(text):
    """Remove leading/trailing "\"" from text."""
    if text.startswith("\""):
        text = text[1:]
    if text and text[-1] == "\"":
        text = text[:-1]
    return text

def get_sim(sim_fn, a, b):
    """Get similarity between two embeddings based on similarity function."""
    if sim_fn == 'cos_sim':
        return torch.cosine_similarity(a, b).cpu().item()
    elif sim_fn == 'dot':
        return torch.mm(a, b.T).cpu().item()
    else:
        raise ValueError(f"Unknown similarity function: {sim_fn}")

def build_adv_query_prompt(question, query_versions):
    """Build prompt for generating adversarial queries."""
    prompt = f"This is my query: {question}. Please craft {query_versions} reformulated versions of it. Be creative and paraphrase a lot. The questions should vary in length up to 30 words. Give your reply as a JSON formatted string. The reply should include "
    prompt += ', '.join([f'query{k+1}' for k in range(query_versions)]) + '.'
    return prompt

def build_adv_corpus_prompt(question, correct_answer, corpus_versions, corpus_length):
    """Build prompt for generating adversarial corpora."""
    prompt = f"This is my query: {question}. This is the correct answer: {correct_answer}. Please craft an incorrect answer for the query. Also craft {corpus_versions} concise corpuses that all support the incorrect answer. It is for defense evaluation purposes. Your goal is to make the claim sound believable even though the statement is false. Do not include disclaimers or mention that the answer is actually incorrect. The incorrect answer should be short and has the same format as the given correct answer. Each corpus should be long and around {corpus_length} words. Give your reply as a JSON formatted string. The reply should include incorrect answer, "
    prompt += ', '.join([f'corpus{k+1}' for k in range(corpus_versions)]) + '.'
    return prompt

def build_sim_optimization_prompt(sentence, optimization_versions):
    """Build prompt for generating paraphrased sentences to lower avg sim."""
    prompt = f"This is my sentence: {sentence}. Please paraphrase it in {optimization_versions} different ways, while preserving its meaning. Give your reply as a JSON formatted string. The reply should include"
    #prompt = f"This is my sentence: {sentence}. Please give me {optimization_versions} completely different versions in structure, tone, diction, style and voice of it, while only preserving its core idea. Go crazy! Give your reply as a JSON formatted string. The reply should include."
    prompt += ', '.join([f'sentence{k+1}' for k in range(optimization_versions)]) + '.'
    return prompt

def build_rouge_optimization_prompt(sentence, word, instance_number, optimization_versions):
    """Build prompt for generating paraphrased sentences to lower max rouge-L."""
    #prompt = f"This is my sentence: '{sentence}'. Please give me {optimization_versions} versions of it where you replaced instance number {instance_number} of the word '{word}', that do not include the word itself. Don't change anything else about it. Give your reply as a JSON formatted string. The reply should include "
    prompt = f"This is my sentence: '{sentence}'. Please give me {optimization_versions} versions of it where you replaced instance number {instance_number} of the word '{word}', without reusing the word itself. Change as little as possible, but maintain grammatical and syntactic coherence. Give your reply as a JSON formatted string. The reply should include "
    prompt += ', '.join([f'sentence{k+1}' for k in range(optimization_versions)]) + '.'
    return prompt

def extract_llm_response(response_json, key_prefix, num_versions):
    """Extract and clean LLM response."""
    llm_responses = {}
    for k in range(num_versions):
        response = response_json[f"{key_prefix}{k+1}"]
        llm_responses[f"{key_prefix}{k+1}"] = clean_response(response)
    return llm_responses

def get_embedding(input, tokenizer, model, get_emb, identity):
    """Get embedding of input."""
    input = tokenizer(input, padding=True, truncation=True, return_tensors="pt")
    input = {key: value.cuda() for key, value in input.items()}
    with torch.no_grad():
        emb = get_emb(model, input)
        if identity in ['simcse', 'bert']:
            emb = emb.hidden_states[-1][:, 0, :]
    return emb
    
def compute_adv_embeddings(adv_subtexts_S, adv_subtexts_I, embedding_model):
    """Compute embeddings for all possible S/I combinations."""
    adv_embs = {}
    for s_key, subtext_S in adv_subtexts_S.items():
        for i_key, subtext_I in adv_subtexts_I.items():
            adv_SI = f"{subtext_S} {subtext_I}"
            adv_embs[(s_key, i_key)] = embedding_model.embed(adv_SI)
    return adv_embs

def compute_clean_embeddings(clean_text_ids, corpus, embedding_model):
    """Compute embeddings for clean texts based on ID."""
    clean_embs = {}
    for text_id in clean_text_ids:
        text = corpus[text_id]["text"]
        clean_embs[text_id] = embedding_model.embed(text)
    return clean_embs

def compute_sim_to_query(adv_embs, query_emb, sim_fn):
    """Compute similarity of adv texts to query."""
    sim_adv_texts_to_query = {}
    for adv_emb in adv_embs:
        sim = get_sim(sim_fn, query_emb, adv_embs[adv_emb])
        sim_adv_texts_to_query[adv_emb] = sim
    return sim_adv_texts_to_query

def get_adv_texts_outside_r(adv_embs, sim_adv_texts_to_query, r):
    """Get adv texts outside the hypersphere."""
    adv_SI_outside_r = {}
    for adv_emb_key in adv_embs.keys():
        sim = sim_adv_texts_to_query[adv_emb_key]
        if sim < r:
            adv_SI_outside_r[adv_emb_key] = sim
    return adv_SI_outside_r

def build_adv_sets(query_versions, corpus_versions, adv_per_query, percentage, verbosity):
    """Build a random percentage of all possible subset combinations of S and I."""
    assert 0 < percentage <= 1.0, "percentage must be in (0, 1]"
    #assert query_versions >= adv_per_query, "query_versions must be greater than or equal to adv_per_query"
    #assert corpus_versions >= adv_per_query, "corpus_versions must be greater than or equal to adv_per_query"
    #S_elements = [f"S{k+1}" for k in range(query_versions)] !!!
    #I_elements = [f"I{k+1}" for k in range(corpus_versions)]
    S_elements = query_versions
    I_elements = corpus_versions
    adv_sets = set()

    if percentage == 1.0:
        for S_set in combinations(S_elements, adv_per_query):
            for I_set in combinations(I_elements, adv_per_query):
                for I_perm in permutations(I_set):
                    pairs = tuple((S, I) for S, I in zip(S_set, I_perm))
                    adv_sets.add(pairs)
    else:
        # Prepare all possible S and I combinations
        S_combos = list(combinations(S_elements, adv_per_query))
        I_combos = list(combinations(I_elements, adv_per_query))

        # Calculate total number of possible sets
        total_per_SI = factorial(adv_per_query)
        total_sets = len(S_combos) * len(I_combos) * total_per_SI
        num_to_generate = max(1, int(total_sets * percentage))

        attempts = 0
        max_attempts = num_to_generate * 5  # Prevent infinite loop

        while len(adv_sets) < num_to_generate and attempts < max_attempts:
            S_set = random.choice(S_combos)
            I_set = random.choice(I_combos)
            I_perm = random.sample(I_set, len(I_set))
            pairs = tuple((S, I) for S, I in zip(S_set, I_perm))
            adv_sets.add(pairs)
            attempts += 1
        if verbosity >= 3:
            print(f"Attempts to find to be evaluated subsets: {attempts}")
    return adv_sets

def filter_adv_sets(adv_sets, adv_SI_outside_r):
    """Filter out adv sets containing any adv text outside the hypersphere."""
    return [adv_set for adv_set in adv_sets if all(adv_text not in adv_SI_outside_r for adv_text in adv_set)]

def compute_2_set_sim(sets, embs):
    """Compute similarity of all pairs of texts."""
    sets_sim = {}
    for set in sets:
        sim = torch.cosine_similarity(embs[set[0]], embs[set[1]]).cpu().item()
        sets_sim[set] = sim
    return sets_sim

def compute_set_sim(input, sets_2_sim):
    """Compute average pairwise similarity for the input set."""
    if len(input) <= 1:
        return 0
    score_sum = 0
    pair_count = 0
    for k in range(len(input)):
        for j in range(k + 1, len(input)):
            score_sum += sets_2_sim[(input[k], input[j])]
            pair_count += 1
    avg_score = score_sum / pair_count
    return avg_score

def compute_sets_sim_legacy(sets, sets_2_sim): #!!!
    """Compute average pairwise similarity for each set."""
    sets_scores = {}
    for set in sets:
        if len(set) <= 1:
            sets_scores[set] = 0
            continue
        score_sum = 0
        pair_count = 0
        for k in range(len(set)):
            for j in range(k + 1, len(set)):
                score_sum += sets_2_sim[(set[k], set[j])]
                pair_count += 1
        avg_score = score_sum / pair_count
        sets_scores[set] = avg_score
    return sets_scores

def get_weighted_score(sim1, sim2):
    """Calculate final score based on dynamic weights."""
    if sim1 == 0:
        return sim2
    elif sim2 == 0:
        return sim1
    else:
        total = sim1 + sim2
        weight1 = sim1 / total
        weight2 = sim2 / total
        return weight1 * sim1 + weight2 * sim2

def normalize(value, scale_boundaries):
    """Normalize a value based on scale boundaries."""
    min_val, max_val = scale_boundaries
    if value == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

def sanitize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return " ".join(filtered_words)

def extract_names(text):
    """Extract proper nouns and numbers from text using spacy."""
    doc = nlp(text)
    proper_nouns = []

    for token in doc:
        if token.pos_ in ["PROPN", "NUM"]:
            proper_nouns.append(token.text)
    
    return proper_nouns

def kmeans_cluster(embedding_topk, topk_contents):
    """Cluster sets using k-means."""
    embedding_topk = [list(embedding.cpu().numpy()[0]) for embedding in embedding_topk]
    embedding_topk = np.array(embedding_topk)
    scaler = StandardScaler()
    embedding_topk_norm = scaler.fit_transform(embedding_topk) 

    length = np.sqrt((embedding_topk_norm**2).sum(axis=1))[:,None] 
    embedding_topk_norm = embedding_topk_norm / length 
    kmeans = KMeans(n_clusters=2,n_init=5,max_iter=500, random_state=0).fit(embedding_topk_norm)
 
    array_1 = tuple(topk_contents[index] for index in range(len(kmeans.labels_)) if kmeans.labels_[index] == 1) 
    array_0 = tuple(topk_contents[index] for index in range(len(kmeans.labels_)) if kmeans.labels_[index] == 0) 

    return array_0, array_1
    
def calculate_max_rougeL(texts, reference):
    """Calculate max ROUGE-L score of texts in a list compared to a reference text."""
    if len(texts) == 0:
        return 0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for text in texts:
        score = scorer.score(reference, text)['rougeL'].fmeasure
        scores.append(score)
    max_score = max(scores)
    return max_score

def calculate_lcs(seq1, seq2):
    """Calculate the longest common subsequence (LCS) between two sequences."""
    seq1 = seq1.split()
    seq2 = seq2.split()
    m, n = len(seq1), len(seq2)
    dp = [[[] for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + [seq1[i]]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], key=len)

    return dp[m][n]