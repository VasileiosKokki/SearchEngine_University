import csv
import math
import os
import pickle
import time
from typing import Dict, List, Callable

import numpy as np
import pandas as pd
import snowballstemmer
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import math

import pytrie
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def load_docs_to_dict(file_path_docs):
    ext = os.path.splitext(file_path_docs)[1].lower()
    delimiter = ',' if ext == '.csv' else '\t'

    # Read file with pandas
    df = pd.read_csv(file_path_docs, delimiter=delimiter, encoding='utf-8', low_memory=False)

    # Add auto-incrementing doc_id column as string "doc_0", "doc_1", ...
    df['doc_id'] = ['doc_' + str(i) for i in range(len(df))]

    docs = {}

    # Iterate rows with progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), unit="lines", desc="Loading docs"):
        # Safety check in case of missing columns
        if len(row) <= 7:
            continue

        doc_id = row['doc_id']
        title = str(row[1]).strip()
        plot = str(row[7]).strip()

        docs[doc_id] = title + " " + plot

    return docs

def load_all_docs_metadata(file_path_docs):
    ext = os.path.splitext(file_path_docs)[1].lower()
    delimiter = ',' if ext == '.csv' else '\t'

    # Read full file with pandas
    df = pd.read_csv(file_path_docs, delimiter=delimiter, encoding='utf-8', low_memory=False)

    # Add auto-increment doc_id as string like "doc_0", "doc_1", ...
    df['doc_id'] = ['doc_' + str(i) for i in range(len(df))]

    docs_metadata = {}

    # Iterate rows efficiently
    for _, row in tqdm(df.iterrows(), total=len(df), unit="lines", desc="Loading docs"):
        # Make sure enough columns (safety)
        if len(row) <= 7:
            continue

        doc_id = row['doc_id']
        release_year = str(row.iloc[0]).strip()
        title = str(row.iloc[1]).strip()
        origin = str(row.iloc[2]).strip()
        director = str(row.iloc[3]).strip()
        cast = str(row.iloc[4]).strip()
        genre = str(row.iloc[5]).strip()
        url = str(row.iloc[6]).strip()
        plot = str(row.iloc[7]).strip()

        docs_metadata[doc_id] = {
            'release_year': release_year,
            'title': title,
            'origin': origin,
            'director': director,
            'cast': cast,
            'genre': genre,
            'url': url,
            'plot': plot,
            'evaluation': title + " " + plot + " " + genre
        }

    return docs_metadata

def load_stopwords():
    stopwords = []
    with open('stopwords.large', 'r', encoding='utf-8') as f:
        for word in f:
            stopwords.append(word.strip().lower())
    return stopwords

def get_top10_results_per_query(queries, doc_ids, model_name,
                           corpus_index=None, stopwords=None, idf_values=None, avgdl=None,
                           sentence_embeddings=None, model=None):
    ranked_results = {}
    start_all = time.time()

    for q_id, q_text in queries.items():
        if model_name == "bm25":
            scores = bm25(
                corpus_index["evaluation"],
                q_text,
                stopwords,
                idf_values["evaluation"],
                avgdl["evaluation"]
            )
        else:
            scores = semantic_similarity(
                doc_ids,
                q_text,
                sentence_embeddings["evaluation"],
                model
            )

        ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:10]
        ranked_results[q_id] = (q_text, ranked_docs)

    total_time = time.time() - start_all

    path = f'dev.3-2-1_{model_name}.qrel'
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            for q_id, (q_text, ranked_docs) in ranked_results.items():
                print(q_id, q_text, ranked_docs)
                for doc_id, _ in ranked_docs:
                    f.write(f"{q_id} {doc_id} 0\n")
    else:
        for q_id, (q_text, ranked_docs) in ranked_results.items():
            print(q_id, q_text, ranked_docs)

    print(f"Total time for all queries for model {model_name} : {total_time:.3f} seconds\n")

    return ranked_results

def merge_and_filter_golden_answers(path1, path2):
    merged_lines = []

    for path in [path1, path2]:
        with open(path, 'r') as file:
            for line in file:
                if line.strip().split()[-1] != '0':
                    merged_lines.append(line)

    with open("golden_answers", 'w') as outfile:
        outfile.writelines(merged_lines)

def save_bm25_representation(stopwords, docs_metadata):
    # Prepare empty dicts to hold tokenized corpora per field
    fields = ['release_year', 'title', 'origin', 'director', 'cast', 'genre', 'url', 'plot', 'evaluation']

    tokenized_corpora = {field: [] for field in fields}
    doc_ids = list(docs_metadata.keys())

    # Tokenize each field's text separately
    for doc_id in tqdm(doc_ids, desc="Tokenizing docs per field"):
        doc = docs_metadata[doc_id]
        for field in fields:
            text = doc.get(field, "")
            tokens = tokenize(text, stopwords)
            tokenized_corpora[field].append(tokens)

    # For each field, build corpus_index (doc_id -> tokens)
    corpus_indexes = {
        field: {doc_id: tokens for doc_id, tokens in zip(doc_ids, tokenized_corpora[field])}
        for field in fields
    }

    # Compute idf_values per field
    idf_values = {}
    for field in tqdm(fields, desc="Computing IDF per field"):
        all_terms = set()
        for tokens in corpus_indexes[field].values():
            all_terms.update(tokens)
        idf_values[field] = {term: compute_idf(term, corpus_indexes[field]) for term in all_terms}

    # Save each corpus_index and idf_values separately
    with open("corpus_index.pkl", "wb") as f:
        pickle.dump(corpus_indexes, f)

    with open("idf_values.pkl", "wb") as f:
        pickle.dump(idf_values, f)

def tokenize(text, stopwords):
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [t for t in tokens if t not in stopwords]

def bm25(corpus, q_text, stopwords, idf_values, avgdl, k1=1.5, b=0.75):
    query_terms = tokenize(q_text, stopwords)
    scores = []
    for doc_id, doc_tokens in corpus.items():
        score = 0.0
        doc_len = len(doc_tokens)
        for term in query_terms:
            tf = doc_tokens.count(term)
            if tf == 0:
                continue  # Term not in document
            idf = idf_values.get(term, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores

def compute_idf(term, corpus):
    N = len(corpus)
    n = sum(1 for doc in corpus if term in doc)
    return np.log((N - n + 0.5) / (n + 0.5) + 1)

def compute_avgdl(corpus_indexes: dict[str, dict[str, list[str]]]) -> dict[str, float]:
    return {
        field: np.mean([len(tokens) for tokens in doc_tokens_by_id.values()])
        for field, doc_tokens_by_id in corpus_indexes.items()
    }

def save_sentence_embeddings(model, docs_metadata):
    fields = ['release_year', 'title', 'origin', 'director', 'cast', 'genre', 'url', 'plot', 'evaluation']
    sentence_embeddings = {}

    for field in fields:

        texts = {doc_id: data[field] for doc_id, data in docs_metadata.items()}

        doc_ids = list(texts.keys())
        values = list(texts.values())

        embeddings_array = model.encode(
            values,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128  # Adjust based on RAM/GPU
        )

        # Map back to doc_ids
        field_embeddings = dict(zip(doc_ids, embeddings_array))
        sentence_embeddings[field] = field_embeddings

    with open(f"sentence_embeddings.pkl", "wb") as f:
        pickle.dump(sentence_embeddings, f)

def semantic_similarity(doc_ids, q_text, embeddings, model):
    query_embedding = model.encode(q_text, convert_to_numpy=True)

    # Get corresponding embeddings
    vecs = [embeddings.get(doc_id) for doc_id in doc_ids]

    # Filter out missing embeddings
    filtered = [
        emb for doc_id, emb in zip(doc_ids, vecs)
        if emb is not None
    ]

    if not filtered:
        return []  # return empty list if no embeddings

    matrix = np.vstack(filtered)

    # Vectorized cosine similarity
    similarities = cosine_similarity([query_embedding], matrix)[0]

    return similarities.tolist()  # return just the scores as a list

def evaluate_topk_results(golden_answers, topk_results, model_name):
    total_precision = 0
    num_queries = len(golden_answers)

    for qid in golden_answers:
        correct_set = set(doc_id for doc_id, _ in golden_answers[qid])  # relevant docs
        _, ranked_docs = topk_results[qid]
        retrieved_set = set(doc_id for doc_id, _ in ranked_docs)        # top-k retrieved docs

        # Compute TP, FP, FN
        tp = len(correct_set & retrieved_set)
        fp = len(retrieved_set - correct_set)

        # Compute Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        total_precision += precision

    # Compute means
    mean_precision = total_precision / num_queries

    # Output results
    print(f"Model: {model_name}")
    print(f"Mean Precision: {mean_precision:.4f}\n")

    return mean_precision
