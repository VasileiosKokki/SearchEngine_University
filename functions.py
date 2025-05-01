import math
import os
import pickle

import numpy as np
import snowballstemmer
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
import re

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import math

import pytrie
from sklearn.metrics.pairwise import cosine_similarity


def load_docs_to_dict(ids_list, file_path_docs):
    docs = {}

    # Open and read the file
    with open(file_path_docs, 'r', encoding='utf-8') as f:
        for line in f:
            # Split each line by tab
            fields = line.strip().split('\t')

            # Assuming the ID is the first field, title is third and abstract is the fourth field
            doc_id = fields[0].strip()
            title = fields[2].strip()
            abstract = fields[3].strip()

            # If the doc_id is in the ids_list, add it to the dictionary
            if doc_id in ids_list:
                docs[doc_id] = title+" "+ abstract

    return docs

def load_all_docs_metadata(file_path_docs):
    docs_metadata = {}

    # Open and read the file
    with open(file_path_docs, 'r', encoding='utf-8') as f:
        for line in f:
            # Split each line by tab
            fields = line.strip().split('\t')

            # Assuming the ID is the first field, URL is the second, title is the third, and abstract is the fourth
            doc_id = fields[0].strip()
            url = fields[1].strip()
            title = fields[2].strip()
            abstract = fields[3].strip()

            # Store the metadata in a dictionary
            docs_metadata[doc_id] = {
                'url': url,
                'title': title,
                'abstract': abstract
            }

    return docs_metadata

def load_stopwords():
    stopwords = []
    with open('stopwords.large', 'r', encoding='utf-8') as f:
        for word in f:
            stopwords.append(word.strip().lower())
    return stopwords

def save_bm25_representation(stopwords):
    with open('dev.docs.ids', 'r', encoding='utf-8') as f:
        # Read the file and strip each line of any leading/trailing whitespace
        docids = [line.strip() for line in f.readlines()]

    documents_dict = load_docs_to_dict(docids, 'doc_dump.txt')

    texts = list(documents_dict.values())
    # Tokenize all documents
    tokenized_corpus = [tokenize(text, stopwords) for text in texts]

    # Map doc_ids to tokenized versions for scoring later
    doc_ids = list(documents_dict.keys())
    corpus_index = {doc_id: tokens for doc_id, tokens in zip(doc_ids, tokenized_corpus)}

    idf_values = {term: compute_idf(term, corpus_index) for term in set(sum(corpus_index.values(), []))}

    # Save the BM25 object
    if not os.path.exists("corpus_index.pkl"):
        with open("corpus_index.pkl", "wb") as f:
            pickle.dump(corpus_index, f)

    if not os.path.exists("idf_values.pkl"):
        with open("idf_values.pkl", "wb") as f:
            pickle.dump(idf_values, f)

# def tokenize(text):
#     return [word.lower() for word in word_tokenize(text)]

def tokenize(text, stopwords):
    pattern = r"^[0-9\+\-\*\·/=–\.\,]+$"
    stemmer = snowballstemmer.stemmer('english')
    tokens = [word for word in nltk.word_tokenize(text) if len(word) > 2 and word not in stopwords and not word.isdigit() and not re.match(pattern, word)]
    # filtered_tokens = [t for t in tokens if t not in stopwords and not t.isdigit() and not re.match(pattern,t)]
    stems = stemmer.stemWords(tokens)
    return stems

def top_n_similar_documents(q_text, stopwords, doc_ids, corpus, idf_values, avgdl):
    tokenized_query = tokenize(q_text, stopwords)
    # scores = bm25.get_scores(tokenized_query)
    scores = bm25(corpus, tokenized_query, idf_values, avgdl)
    ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:10]
    return ranked_docs

def bm25(corpus, query_terms, idf_values, avgdl, k1=1.5, b=0.75):
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

def compute_avgdl(corpus):
    return np.mean([len(doc_tokens) for doc_tokens in corpus.values()])
