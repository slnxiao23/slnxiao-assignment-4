from flask import Flask, request, jsonify, send_from_directory
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load dataset
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(newsgroups.data)

# Apply SVD for dimensionality reduction (LSA)
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

def process_query(query):
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)
    similarities = cosine_similarity(query_reduced, X_reduced)
    top_5_indices = np.argsort(similarities[0])[::-1][:5]
    return top_5_indices, similarities[0][top_5_indices]

# Serve the index.html file
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Endpoint to handle search requests
@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    top_indices, similarities = process_query(query)
    results = [{'document': newsgroups.data[i], 'similarity': similarities[idx]} for idx, i in enumerate(top_indices)]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
