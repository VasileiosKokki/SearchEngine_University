import os

import faiss
import numpy as np

def save_faiss_indexes(embeddings_by_field):
    os.makedirs("faiss_bundle", exist_ok=True)
    for field, embedding_dict in embeddings_by_field.items():
        matrix = np.vstack(list(embedding_dict.values())).astype(np.float32)

        faiss.normalize_L2(matrix)
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        index_path = os.path.join("faiss_bundle", f"{field}.faiss")
        faiss.write_index(index, index_path)


def load_faiss_indexes():
    faiss_indexes = {}
    for filename in os.listdir("faiss_bundle"):
        if filename.endswith(".faiss"):
            field = filename.replace(".faiss", "")
            path = os.path.join("faiss_bundle", filename)
            faiss_indexes[field] = faiss.read_index(path)
    return faiss_indexes