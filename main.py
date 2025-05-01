from functions import *

def startup():

    # ---------- 1) Define representations of the texts ----------

    stopwords = load_stopwords()

    # one time only, then we simply open
    # save_bm25_representation()

    # takes a lot of time
    with open("corpus_index.pkl", "rb") as file:
        corpus_index = pickle.load(file)

    with open("idf_values.pkl", "rb") as file:
        idf_values = pickle.load(file)

    # ---------- Queries ----------
    # # ACTION: open the queries file and read three queries
    queries = {}
    with open('../dev.titles.queries', 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split('\t')
            q_id = fields[0].strip()
            q_text = fields[1].strip()

            if q_id in ['PLAIN-1', 'PLAIN-11', 'PLAIN-111']:
                queries[q_id] = q_text

    # ---------- 2) Implement an indexing mechanism ----------

    # trie = build_trie_index(tfidf_representation)

    # ---------- 3) Implement the mechanism for retrieving and calibrating the results ----------

    doc_ids = list(corpus_index.keys())
    avgdl = compute_avgdl(corpus_index)

    for q_id, q_text in queries.items():
        ranked_docs = top_n_similar_documents(q_text, stopwords, doc_ids, corpus_index, idf_values, avgdl)
        print(q_id, queries[q_id], ranked_docs)

    # return vectorizer, tfidf_representation, trie, vocabulary
    return corpus_index, idf_values, doc_ids, stopwords, avgdl