import pickle

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def save_sentence_embeddings(model, docs_metadata):
    fields = ['title', 'origin', 'director', 'cast', 'genre', 'plot', 'evaluation']
    sentence_embeddings = {}

    for field in fields:

        texts = {doc_id: data[field] for doc_id, data in docs_metadata.items() if not pd.isna(data[field])}

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

def semantic_similarity(q_text, embeddings, model, index, use_index=True):
    # Encode query
    query_embedding = model.encode(q_text, convert_to_numpy=True)

    # Extract doc_ids and their embeddings directly
    doc_ids = list(embeddings.keys())
    matrix = np.vstack(list(embeddings.values()))

    if use_index:
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        faiss.normalize_L2(query_embedding)

        scores, idxs = index.search(query_embedding, k=10)

        ranked_docs = [(doc_ids[i], float(scores[0][j])) for j, i in enumerate(idxs[0])][:10]
        return ranked_docs
    else:
        # Fallback: cosine similarity without FAISS
        similarities = cosine_similarity([query_embedding], matrix)[0]
        scored_docs = list(zip(doc_ids, similarities.tolist()))
        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:10]
        return ranked_docs