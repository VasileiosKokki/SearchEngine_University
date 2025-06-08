import time

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

from helper_funcs.judge_and_golden_answers import judge_single_query
from main import startup
from models.bm25 import tokenized_similarity
from models.sentence_transformer import semantic_similarity

app = Flask(__name__)

with app.app_context():
    corpus_index, idf_values, doc_ids, stopwords, avgdl, sentence_embeddings, transformer_model, docs_metadata, api_key, faiss_indexes = startup()
    print("Flask app startup complete.")  # Debugging line to confirm the startup


@app.route('/')
def home():
    return send_from_directory(".", "client.html")
@app.route('/search', methods=['GET'])
def search():
    user_query = request.args.get('query')  # Get ?query= from URL
    field = request.args.get('field')
    use_semantic_search = request.args.get('useSemanticSearch') == 'true'

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Tokenize the query
    if use_semantic_search:
        ranked_docs = semantic_similarity(user_query, sentence_embeddings[field], transformer_model, faiss_indexes[field], True)
    else:
        ranked_docs = tokenized_similarity(corpus_index[field], user_query, stopwords, idf_values[field], avgdl[field])

    # Format the response with metadata
    docs_with_meta = []
    for doc_id, _ in ranked_docs:
        metadata = docs_metadata.get(doc_id, None)  # Get the metadata for the document
        if metadata:
            docs_with_meta.append({
                'doc_id': doc_id,
                'release_year': metadata['release_year'] if pd.notna(metadata['release_year']) else 'unknown',
                'title': metadata['title'] if pd.notna(metadata['title']) else 'unknown',
                'origin': metadata['origin'] if pd.notna(metadata['origin']) else 'unknown',
                'director': metadata['director'] if pd.notna(metadata['director']) else 'unknown',
                'cast': metadata['cast'] if pd.notna(metadata['cast']) else 'unknown',
                'genre': metadata['genre'] if pd.notna(metadata['genre']) else 'unknown',
                'url': metadata['url'] if pd.notna(metadata['url']) else 'unknown',
                'plot': metadata['plot'] if pd.notna(metadata['plot']) else 'unknown',
                'evaluation': metadata['evaluation'] if pd.notna(metadata['evaluation']) else 'unknown'
            })


    # Return the results
    return jsonify({'results': docs_with_meta})

@app.route('/judge', methods=['POST'])
def judge():
    data = request.get_json()

    user_query = data.get('query')
    docs_with_eval_meta = data.get('docs_with_eval_meta')

    judged_docs = judge_single_query(user_query, docs_with_eval_meta, api_key)

    return judged_docs

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
