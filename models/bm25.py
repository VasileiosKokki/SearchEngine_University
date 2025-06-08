import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def save_bm25_representation(stopwords, docs_metadata):
    # Prepare empty dicts to hold tokenized corpora per field
    fields = ['title', 'origin', 'director', 'cast', 'genre', 'plot', 'evaluation']

    tokenized_corpora = {field: {} for field in fields}
    doc_ids = list(docs_metadata.keys())

    # Tokenize each field's text separately
    for doc_id in tqdm(doc_ids, desc="Tokenizing docs per field"):
        doc = docs_metadata[doc_id]
        for field in fields:
            text = doc[field]
            if not pd.isna(text):
                tokens = tokenize(text, stopwords)
                tokenized_corpora[field][doc_id] = tokens

    # For each field, build corpus_index
    corpus_indexes = {
        field: doc_tokens
        for field, doc_tokens in tokenized_corpora.items()
    }

    # Compute idf_values per field
    idf_values = {}
    for field in tqdm(fields, desc="Computing IDF per field"):
        if field in corpus_indexes:
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

def tokenized_similarity(corpus, q_text, stopwords, idf_values, avgdl, k1=1.5, b=0.75):
    query_terms = tokenize(q_text, stopwords)
    scored_docs = []
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
        scored_docs.append((doc_id, score))

    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:10]
    return ranked_docs

def compute_idf(term, corpus):
    N = len(corpus)
    n = sum(1 for doc in corpus if term in doc)
    return np.log((N - n + 0.5) / (n + 0.5) + 1)

def compute_avgdl(corpus_indexes: dict[str, dict[str, list[str]]]) -> dict[str, float]:
    return {
        field: np.mean([len(tokens) for tokens in doc_tokens_by_id.values()])
        for field, doc_tokens_by_id in corpus_indexes.items()
    }