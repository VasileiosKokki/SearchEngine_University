import math

import pytrie

def top_n_similar_documents(query_vector, documents_dict, n=10):
    """
    Find the top N documents with maximum cosine similarity to the query vector.

    Parameters:
    - query_vector: The vector representation of the query document.
    - documents_dict: A dictionary where keys are document IDs and values are vectors.
    - n: Number of top documents to return.

    Returns:
    - A list of tuples (doc_id, similarity_score), sorted by similarity in descending order.
    """
    similarities = []
    for doc_id, doc_vector in documents_dict.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        similarities.append((doc_id, similarity))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Return top N
    return similarities[:n]

def cosine_similarity(doc1, doc2):
    """
    Computes the cosine similarity between two documents represented as weighted dictionaries.

    Parameters:
        doc1 (dict): The first document with terms as keys and weights as values.
        doc2 (dict): The second document with terms as keys and weights as values.

    Returns:
        float: The cosine similarity between the two documents (range: 0 to 1).
    """
    # Compute the dot product of the two documents
    dot_product = sum(doc1.get(term, 0) * doc2.get(term, 0) for term in set(doc1) | set(doc2))

    # Compute the magnitude of each document
    magnitude_doc1 = math.sqrt(sum(weight ** 2 for weight in doc1.values()))
    magnitude_doc2 = math.sqrt(sum(weight ** 2 for weight in doc2.values()))

    # Handle edge case: if one of the documents is zero vector, similarity is 0
    if magnitude_doc1 == 0 or magnitude_doc2 == 0:
        return 0.0

    # Compute the cosine similarity
    similarity = dot_product / (magnitude_doc1 * magnitude_doc2)
    return similarity


def build_trie_index(documents):
    trie = pytrie.StringTrie()
    for doc_id, term_weights in documents.items():
        for term in term_weights.keys():
            if term_weights[term]!=0:
                if term in trie:
                    trie[term].append(doc_id)
                else:
                    trie[term]=[doc_id]
    return trie


def get_union_doc_ids(non_zero_terms1, inverted_index):
    doc_ids = set()
    for term, weight in non_zero_terms1.items():
        if term in inverted_index:
            doc_ids.update(inverted_index[term])
    return doc_ids

def get_intersection_doc_ids(non_zero_terms1, inverted_index):
    doc_ids = set()
    for term, weight in non_zero_terms1.items():
        if term in inverted_index:
            term_doc_ids = set(inverted_index[term])
            if doc_ids is None:
                doc_ids = term_doc_ids
            else:
                doc_ids.intersection_update(term_doc_ids)
    return doc_ids if doc_ids is not None else set()

def get_docs_with_half_terms(non_zero_terms1, inverted_index):
    required_count = len(non_zero_terms1) // 2  # Minimum number of terms a document must match
    doc_count = {}  # Dictionary to count matches per document

    for term in non_zero_terms1:
        if term in inverted_index:
            for doc_id in inverted_index[term]:
                doc_count[doc_id] = doc_count.get(doc_id, 0) + 1

    # Filter documents meeting the required count
    result_docs = {doc_id for doc_id, count in doc_count.items() if count >= required_count}

    return result_docs

def get_top_docs_using_trie_and_union(query, alldocs, inverted_index, numresults):
    non_zero_terms = {term: weight for term, weight in query.items() if weight != 0.0}
    doc_ids = get_union_doc_ids(non_zero_terms,inverted_index)
    print(len(doc_ids), len(alldocs))
    selected_docs={doc_id: alldocs[doc_id] for doc_id in doc_ids if doc_id in alldocs}
    return top_n_similar_documents(query, selected_docs, n=numresults)


def get_top_docs_using_trie_and_intersection(query, alldocs, inverted_index, numresults):
    non_zero_terms = {term: weight for term, weight in query.items() if weight != 0.0}
    doc_ids = get_intersection_doc_ids(non_zero_terms,inverted_index)
    print(len(doc_ids), len(alldocs))
    selected_docs={doc_id: alldocs[doc_id] for doc_id in doc_ids if doc_id in alldocs}
    return top_n_similar_documents(query, selected_docs, n=numresults)


def get_top_docs_using_trie_and_half(query, alldocs, inverted_index, numresults):
    non_zero_terms = {term: weight for term, weight in query.items() if weight != 0.0}
    doc_ids = get_docs_with_half_terms(non_zero_terms,inverted_index)
    print(len(doc_ids), len(alldocs))
    selected_docs={doc_id: alldocs[doc_id] for doc_id in doc_ids if doc_id in alldocs}
    return top_n_similar_documents(query, selected_docs, n=numresults)