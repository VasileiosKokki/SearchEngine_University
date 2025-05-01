# server.py
import time

from flask import Flask, request, jsonify, send_from_directory

from functions import load_all_docs_metadata, top_n_similar_documents
from main import startup

app = Flask(__name__)

with app.app_context():
    # vectorizer, tfidf_representation, trie, vocabulary = startup()  # Call the startup function to initialize the global variables
    corpus_index, idf_values, doc_ids, stopwords, avgdl = startup()
    docs_metadata = load_all_docs_metadata('doc_dump.txt')
    print("Flask app startup complete.")  # Debugging line to confirm the startup


@app.route('/')
def home():
    return send_from_directory(".", "index.html")
@app.route('/search', methods=['GET'])
def search():
    start_time = time.time()  # Start timing
    user_query = request.args.get('query')  # Get ?query= from URL

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Tokenize the query
    ranked_docs = top_n_similar_documents(user_query, stopwords, doc_ids, corpus_index, idf_values, avgdl)

    # Format the response with metadata
    results = []
    for doc_id, score in ranked_docs:
        metadata = docs_metadata.get(doc_id, None)  # Get the metadata for the document
        if metadata:
            results.append({
                'doc_id': doc_id,
                'title': metadata['title'],
                'abstract': metadata['abstract'],
                'url': metadata['url']
            })


    end_time = time.time()  # End timing
    duration = end_time - start_time
    print(duration)
    # Return the results
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
