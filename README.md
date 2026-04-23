# 🎬 Cinematic AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/LLaMA_3.1-Groq-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/BERT-MiniLM-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RAG-Pipeline-green?style=for-the-badge"/>
</p>

A conversational AI movie recommender built on a RAG pipeline — BERT retrieves, LLaMA selects, Flask serves.

---

## How It Works

```
User Query
    │
    ▼
Context Builder  ──  expands short queries using last 3 conversation turns
    │
    ▼
BERT Retrieval   ──  all-MiniLM-L6-v2 → cosine similarity → top-30 candidates
    │
    ▼
LLaMA 3.1        ──  grounded to top-30 only → JSON { reply, movies[] }
    │
    ▼
TMDB Posters     ──  10 parallel threads → poster URLs
    │
    ▼
JSON Response    ──  { reply, title, reason, poster }
```

> **Core design principle:** The LLM only picks from BERT-retrieved movies. It cannot hallucinate a movie that doesn't exist.

---

## Stack

| Layer | Tech |
|-------|------|
| Web | Flask |
| Embeddings | `sentence-transformers` all-MiniLM-L6-v2 |
| LLM | Groq API — llama-3.1-8b-instant |
| Recommender | scikit-learn cosine similarity |
| Posters | TMDB API + ThreadPoolExecutor |
| Data | TMDB 5000 + MovieLens |

---

## Recommendation Modes

- **Chatbot (RAG)** — semantic search + LLM, multi-turn context
- **Content-Based** — CountVectorizer tags + cosine similarity
- **Hybrid** — Content 60% + Collaborative Filtering 40% (cold-start safe)

---

## Setup

```bash
git clone https://github.com/yourusername/cinematic-ai.git
pip install -r requirements.txt
```

Add `.env`:
```
TMDB_API_KEY=your_key
GROQ_API_KEY=your_key
```

Run the notebook to generate `.pkl` files, then:
```bash
python app.py
```

---

## Project Structure

```
cinematic-ai/
├── app.py                             # Flask routes + RAG pipeline
├── movie_recommendation_system.ipynb  # ML training notebook
├── templates/
│   ├── home.html                      # Popular movies
│   ├── chat.html                      # AI chatbot
│   └── recommend.html                 # Dropdown recommender
└── *.pkl                              # Serialized model artifacts
```
