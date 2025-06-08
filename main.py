import os
import pickle
import time

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from helper_funcs.indexing import load_faiss_indexes, save_faiss_indexes
from helper_funcs.judge_and_golden_answers import retrieve_golden_answers, save_golden_answers, merge_queries_top10_judged, judge_single_query
from models.bm25 import save_bm25_representation, compute_avgdl, tokenized_similarity
from models.sentence_transformer import save_sentence_embeddings, semantic_similarity
from helper_funcs.evaluation import evaluate_topk_results
from helper_funcs.loading_and_preprocessing import load_stopwords, load_all_docs_metadata

# dont run this file, run server.py that will also run this

def startup():

    # ---------- 1) Loading and Preprocessing ----------

    stopwords = load_stopwords()
    docs_metadata = load_all_docs_metadata('wiki_movie_plots_deduped.csv')

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device=device)

    load_dotenv()
    api_key = os.getenv("HF_API_KEY")

    #Queries
    queries = {}
    with open('dev.titles.queries', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            q_id = f"Q{i+1}"
            q_text = line.strip()
            queries[q_id] = q_text



    # ---------- 2) Define representations of the texts ----------

    #bm25
    if not os.path.exists("corpus_index.pkl") or not os.path.exists("idf_values.pkl"):
        save_bm25_representation(stopwords, docs_metadata)

    with open("corpus_index.pkl", "rb") as file:
        corpus_index = pickle.load(file)
        avgdl = compute_avgdl(corpus_index)

    with open("idf_values.pkl", "rb") as file:
        idf_values = pickle.load(file)


    #sbert_representations
    if not os.path.exists("sentence_embeddings.pkl"):
        save_sentence_embeddings(model, docs_metadata)

    with open("sentence_embeddings.pkl", "rb") as file:
        sentence_embeddings = pickle.load(file)



    # ---------- 3) Implement an indexing mechanism ----------

    if not os.path.exists("faiss_bundle"):
        save_faiss_indexes(sentence_embeddings)

    faiss_indexes = load_faiss_indexes()



    # ---------- 4) Implement the mechanism for retrieving and ranking the results ----------

    doc_ids = list(docs_metadata.keys())

    field = 'evaluation'

    #sbert_representations - without index
    queries_top_10_docs_sbert = {}
    start_all = time.time()
    for q_id, q_text in queries.items():
        ranked_docs = semantic_similarity(
            q_text=q_text,
            embeddings=sentence_embeddings[field],
            model=model,
            index=faiss_indexes[field],
            use_index=False
        )
        queries_top_10_docs_sbert[q_id] = (q_text, ranked_docs)

    total_time_sbert = time.time() - start_all
    print(f"\nModel: {model_name}")
    for q_id, (q_text, ranked_docs) in queries_top_10_docs_sbert.items():
        print(q_id, q_text, ranked_docs)


    #sbert_representations - with index
    queries_top_10_docs_sbert = {}
    start_all = time.time()
    for q_id, q_text in queries.items():
        ranked_docs = semantic_similarity(
            q_text=q_text,
            embeddings=sentence_embeddings[field],
            model=model,
            index=faiss_indexes[field],
            use_index=True
        )
        queries_top_10_docs_sbert[q_id] = (q_text, ranked_docs)

    total_time_sbert_with_index = time.time() - start_all
    print(f"\nModel: {model_name} with Index")
    for q_id, (q_text, ranked_docs) in queries_top_10_docs_sbert.items():
        print(q_id, q_text, ranked_docs)


    #bm25
    queries_top_10_docs_bm25 = {}
    start_all = time.time()
    for q_id, q_text in queries.items():
        ranked_docs = tokenized_similarity(
            q_text=q_text,
            corpus=corpus_index[field],
            stopwords=stopwords,
            idf_values=idf_values[field],
            avgdl=avgdl[field]
        )
        queries_top_10_docs_bm25[q_id] = (q_text, ranked_docs)

    total_time_bm25 = time.time() - start_all
    print(f"\nModel: bm25")
    for q_id, (q_text, ranked_docs) in queries_top_10_docs_bm25.items():
        print(q_id, q_text, ranked_docs)



    # ---------- 5) Evaluate the time needed and the relevance of the results ----------

    if not os.path.exists("golden_answers.txt"):
        judged_sbert = {}
        for q_id, (q_text, ranked_docs) in tqdm(queries_top_10_docs_sbert.items(), desc="Getting golden answers from LLM"):
            docs_with_eval_meta = []
            for doc_id, _ in ranked_docs:
                metadata = docs_metadata.get(doc_id, None)  # Get the metadata for the document
                if metadata:
                    docs_with_eval_meta.append({
                        'doc_id': doc_id,
                        'evaluation': metadata['evaluation'],
                    })
            result = judge_single_query(q_text, docs_with_eval_meta, api_key)
            judged_sbert[q_id] = (q_text, result)

        judged_bm25 = {}
        for q_id, (q_text, ranked_docs) in tqdm(queries_top_10_docs_bm25.items(), desc="Getting golden answers from LLM"):
            docs_with_eval_meta = []
            for doc_id, _ in ranked_docs:
                metadata = docs_metadata.get(doc_id, None)  # Get the metadata for the document
                if metadata:
                    docs_with_eval_meta.append({
                        'doc_id': doc_id,
                        'evaluation': metadata['evaluation'],
                    })
            result = judge_single_query(q_text, docs_with_eval_meta, api_key)
            judged_bm25[q_id] = (q_text, result)


        merged_judged = merge_queries_top10_judged(judged_sbert, judged_bm25)
        save_golden_answers(merged_judged)


    golden_answers = retrieve_golden_answers()

    mean_precision_sbert = evaluate_topk_results(golden_answers, queries_top_10_docs_sbert)
    mean_precision_bm25 = evaluate_topk_results(golden_answers, queries_top_10_docs_bm25)


    print(f"\nModel: {model_name}")
    print(f"Mean Precision: {mean_precision_sbert:.4f}")
    print(f"Total time for all queries for model {model_name} : {total_time_sbert:.3f} seconds")
    print(f"Total time for all queries for model {model_name} with index : {total_time_sbert_with_index:.3f} seconds")

    print(f"\nModel: bm25")
    print(f"Mean Precision: {mean_precision_bm25:.4f}")
    print(f"Total time for all queries for model bm25 : {total_time_bm25:.3f} seconds")



    return corpus_index, idf_values, doc_ids, stopwords, avgdl, sentence_embeddings, model, docs_metadata, api_key, faiss_indexes