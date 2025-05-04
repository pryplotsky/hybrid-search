import faiss
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('punkt')  # Ensure the standard punkt tokenizer is also available


# Sample Documents
documents = [
    "Machine learning is a field of artificial intelligence.",
    "Natural language processing enables computers to understand text.",
    "FAISS is a library for efficient similarity search in high-dimensional spaces.",
    "BM25 is a ranking function used in search engines.",
    "Deep learning has revolutionized AI and NLP.",
]

# Tokenize and preprocess for BM25
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Load SBERT Model for Semantic Search
model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(documents, convert_to_numpy=True)

# Normalize embeddings (helps FAISS retrieval accuracy)
document_embeddings = normalize(document_embeddings, axis=1, norm='l2')

# Build FAISS index
embedding_dim = document_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(document_embeddings)


def hybrid_search(query, top_k=3):
    """Perform hybrid search with BM25 and FAISS."""
    # BM25 keyword search
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_rankings = np.argsort(bm25_scores)[::-1][:top_k]

    # Semantic search using FAISS
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1, norm='l2')
    _, faiss_rankings = index.search(query_embedding, top_k)
    faiss_rankings = faiss_rankings[0]

    # Combine scores (weighted sum approach)
    combined_results = {}
    for rank, idx in enumerate(bm25_rankings):
        combined_results[idx] = combined_results.get(idx, 0) + (1 / (rank + 1))
    for rank, idx in enumerate(faiss_rankings):
        combined_results[idx] = combined_results.get(idx, 0) + (1 / (rank + 1))

    # Sort by final score
    ranked_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    return [(documents[idx], score) for idx, score in ranked_results]


# Example Query
query = "artificial intelligence and deep learning"
results = hybrid_search(query)

# Print Results
for doc, score in results:
    print(f"Score: {score:.4f} | {doc}")
