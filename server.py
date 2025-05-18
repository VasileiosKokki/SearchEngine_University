# server.py
import time

from flask import Flask, request, jsonify, send_from_directory

from functions import load_all_docs_metadata, bm25, semantic_similarity
from main import startup

app = Flask(__name__)

with app.app_context():
    # vectorizer, tfidf_representation, trie, vocabulary = startup()  # Call the startup function to initialize the global variables
    corpus_index, idf_values, doc_ids, stopwords, avgdl, sentence_embeddings, transformer_model, docs_metadata = startup()
    print("Flask app startup complete.")  # Debugging line to confirm the startup


@app.route('/')
def home():
    return send_from_directory(".", "index.html")
@app.route('/search', methods=['GET'])
def search():
    start_time = time.time()  # Start timing
    user_query = request.args.get('query')  # Get ?query= from URL
    search_by = request.args.get('searchBy')
    use_semantic_search = request.args.get('useSemanticSearch') == 'true'

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Tokenize the query
    if use_semantic_search:
        scores = semantic_similarity(doc_ids, user_query, sentence_embeddings[search_by], transformer_model)
    else:
        scores = bm25(corpus_index[search_by], user_query, stopwords, idf_values[search_by], avgdl[search_by])

    ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:10]

    # Format the response with metadata
    results = []
    for doc_id, score in ranked_docs:
        metadata = docs_metadata.get(doc_id, None)  # Get the metadata for the document
        if metadata:
            results.append({
                'doc_id': doc_id,
                'release_year': metadata['release_year'],
                'title': metadata['title'],
                'origin': metadata['origin'],
                'director': metadata['director'],
                'cast': metadata['cast'],
                'genre': metadata['genre'],
                'url': metadata['url'],
                'plot': metadata['plot'],
            })


    end_time = time.time()  # End timing
    duration = end_time - start_time
    print(duration)
    # Return the results
    return jsonify({'results': results, 'time_needed': duration})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
