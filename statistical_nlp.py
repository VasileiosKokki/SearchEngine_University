import math
import pickle
import time

import nltk
import re
import snowballstemmer
from tqdm import tqdm

from functions import *
from indexing_retrieval import *
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


with open('dev.docs.ids', 'r', encoding='utf-8') as f:
    # Read the file and strip each line of any leading/trailing whitespace
    docids = [line.strip() for line in f.readlines()]

documents_dict = load_docs_to_dict(docids, 'doc_dump.txt')

# len(documents_dict)
# first_value = list(documents_dict.values())[0]
# print(first_value)

stopwords = []
with open('stopwords.large', 'r', encoding='utf-8') as f:
    for word in f:
        stopwords.append(word.strip().lower())


# nltk.download('punkt')  # Download necessary resources
# nltk.download('all')

pattern = r"^[0-9\+\-\*\·/=–\.\,]+$"

stemmer = snowballstemmer.stemmer('english')

texts = list(documents_dict.values())


# ---------- 1) Define representations of the texts ----------
# ACTION: Get tf_idf vector representations and the vocabulary using TfidfVectorizer

# one time only, then we simply open
# save_vectorizer_tfidf_representations(texts, documents_dict)

# takes a lot of time
with open("document_representations.pkl", "rb") as file:
    tfidf_representation = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

vocabulary = vectorizer.get_feature_names_out()

# ---------- Queries ----------
# # ACTION: open the queries file and read three queries
queries = {}
with open('dev.titles.queries', 'r', encoding='utf-8') as f:
    for line in f:
        fields = line.strip().split('\t')
        q_id = fields[0].strip()
        q_text = fields[1].strip()

        if q_id in ['PLAIN-1', 'PLAIN-11', 'PLAIN-111']:
            queries[q_id] = q_text

print(queries)

# # ACTION: Get the tf_idf query representations
tfidf_queries_matrix = vectorizer.transform(list(queries.values()))
tfidf_query_array = tfidf_queries_matrix.toarray()
tfidf_query_representation = {}
for q_id, tfidf in zip(queries.keys(), tfidf_query_array):
    tfidf_query_representation[q_id] = dict(zip(vocabulary, tfidf))

# tfidf_query_representation['PLAIN-1']

# ---------- 2) Implement an indexing mechanism ----------

trie = build_trie_index(tfidf_representation)

# ---------- 3) Implement the mechanism for retrieving and calibrating the results ----------

# top_similar1 = top_n_similar_documents(tfidf_query_representation['PLAIN-1'], tfidf_representation, n=10)
# print(top_similar1)
# top_similar11 = top_n_similar_documents(tfidf_query_representation['PLAIN-11'], tfidf_representation, n=10)
# print(top_similar11)
# top_similar111 = top_n_similar_documents(tfidf_query_representation['PLAIN-111'], tfidf_representation, n=10)
# print(top_similar111)

top_similar1 = get_top_docs_using_trie_and_half(tfidf_query_representation['PLAIN-1'], tfidf_representation, trie, 10)
top_similar11 = get_top_docs_using_trie_and_half(tfidf_query_representation['PLAIN-11'], tfidf_representation, trie, 10)
top_similar111 = get_top_docs_using_trie_and_half(tfidf_query_representation['PLAIN-111'], tfidf_representation, trie, 10)
print(top_similar1)
print(top_similar11)
print(top_similar111)