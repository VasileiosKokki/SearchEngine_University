# server.py
from flask import Flask, request, jsonify, send_from_directory

from functions import load_all_docs_metadata
from statistical_nlp import startup
from indexing_retrieval import get_top_docs_using_trie_and_half

app = Flask(__name__)

with app.app_context():
    vectorizer, tfidf_representation, trie, vocabulary = startup()  # Call the startup function to initialize the global variables
    docs_metadata = load_all_docs_metadata('doc_dump.txt')
    print("Flask app startup complete.")  # Debugging line to confirm the startup


@app.route('/')
def home():
    return send_from_directory(".", "index.html")
@app.route('/search', methods=['GET'])
def search():
    user_query = request.args.get('query')  # Get ?query= from URL

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # TF-IDF vectorization
    query_vector = vectorizer.transform([user_query]).toarray()[0]  # 1D array

    # Build {term: weight} dict, but only for non-zero terms
    query_dict = {term: weight for term, weight in zip(vocabulary, query_vector) if weight != 0.0}

    # Get top documents
    top_docs = get_top_docs_using_trie_and_half(query_dict, tfidf_representation, trie, numresults=10)

    # Extract the document IDs from the top results
    top_doc_ids = [doc_id for doc_id, _ in top_docs]

    # Format the response with metadata
    results = []
    for doc_id in top_doc_ids:
        metadata = docs_metadata.get(doc_id, None)  # Get the metadata for the document
        if metadata:
            results.append({
                'doc_id': doc_id,
                'title': metadata['title'],
                'abstract': metadata['abstract'],
                'url': metadata['url']
            })

    # Return the results
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
