import math
import os
import pickle
import re

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


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


# def tokenize(text, pattern, stopwords, stemmer):
#     tokens = [word for word in nltk.word_tokenize(text) if len(word) > 2 and word not in stopwords and not word.isdigit() and not re.match(pattern, word)]
#     # filtered_tokens = [t for t in tokens if t not in stopwords and not t.isdigit() and not re.match(pattern,t)]
#     stems = stemmer.stemWords(tokens)
#     return stems


def save_vectorizer_tfidf_representations(texts, documents_dict):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()

    tfidf_array = tfidf_matrix.toarray()
    tfidf_representation = {}
    for doc_id, tfidf in zip(documents_dict.keys(), tfidf_array):
        tfidf_representation[doc_id] = dict(zip(vocabulary, tfidf))

    # Save the TF-IDF representation dictionary

    if not os.path.exists("document_representations.pkl"):
        with open("document_representations.pkl", "wb") as file:
            pickle.dump(tfidf_representation, file)

    # Save the vectorizer
    if not os.path.exists("vectorizer.pkl"):
        with open("vectorizer.pkl", "wb") as file:
            pickle.dump(vectorizer, file)