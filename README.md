# HybridSearchAI
# 🔍 HybridSearchAI: Intelligent Document Retrieval with BM25 and Semantic Embeddings

**HybridSearchAI** is a hybrid document search system that combines the strengths of traditional keyword-based retrieval (BM25) and modern semantic search using Sentence-BERT embeddings and FAISS. This project demonstrates how combining symbolic and neural techniques can lead to more accurate and context-aware search results in small to mid-scale text corpora.

---

## 🚀 Features

- ✅ **BM25 keyword relevance** using `rank_bm25`
- ✅ **Semantic search** using Sentence-BERT (`all-MiniLM-L6-v2`)
- ✅ **Fast vector similarity** using `FAISS` for efficient search
- ✅ **Score fusion** from both methods for improved result ranking
- ✅ Easy to adapt for larger text datasets

---

## 🧠 How It Works

1. **Input documents** are preprocessed and embedded using SBERT.
2. **BM25** is used to score documents based on tokenized keyword relevance.
3. **FAISS** performs vector similarity search over SBERT embeddings.
4. Final ranking is based on a **weighted fusion** of both ranking strategies.

---
