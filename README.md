# Cinematic AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/LLaMA_3.1-Groq-6236FF?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/BERT-MiniLM-FF6B35?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAG-Pipeline-00C896?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TMDB-API-01B4E4?style=for-the-badge&logo=themoviedatabase&logoColor=white"/>
</p>

<p align="center">
  <strong>A conversational AI movie agent — BERT retrieves, LLaMA selects, Flask serves.</strong>
</p>


## How It Works

```
User Query
    │
    ▼
Context Builder  ──  expands short queries using last 3 conversation turns
    │
    ▼
BERT Retrieval   ──  all-MiniLM-L6-v2 · cosine similarity · top-30 candidates
    │
    ▼
LLaMA 3.1        ──  grounded to top-30 only · returns JSON { reply, movies[] }
    │
    ▼
TMDB Posters     ──  10 parallel threads · real-time poster URLs
    │
    ▼
JSON Response    ──  { reply, title, reason, poster }
```

> **Core design principle:** The LLM is grounded to BERT-retrieved results only.
> It cannot hallucinate a movie that doesn't exist in the database.

---

## Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| Semantic Search | `sentence-transformers` · all-MiniLM-L6-v2 |
| LLM Inference | Groq API · llama-3.1-8b-instant |
| Similarity Engine | scikit-learn · cosine similarity |
| Poster Fetching | TMDB API · ThreadPoolExecutor (10 workers) |
| Training Data | TMDB 5000 · MovieLens |

---

## Recommendation Modes

| Mode | How |
|---|---|
| Chatbot (RAG) | BERT semantic search + LLM · multi-turn context aware |
| Content-Based | CountVectorizer tags + cosine similarity matrix |
| Hybrid | Content 60% + Collaborative Filtering 40% · cold-start safe |

---

## Setup

```bash
git clone https://github.com/OwaisZafar01/Personalized-Recommendation-System.git
cd Personalized-Recommendation-System
pip install -r requirements.txt
```

Create a `.env` file in the root:

```env
TMDB_API_KEY=your_key
GROQ_API_KEY=your_key
```

Run the training notebook to generate `.pkl` artifacts, then start the server:

```bash
python app.py
```

---

## Project Structure

```
Personalized-Recommendation-System/
├── app.py                             # Flask routes + RAG pipeline
├── movie_recommendation_system.ipynb  # ML training notebook
├── templates/
│   ├── home.html                      # Popular movies homepage
│   ├── chat.html                      # AI chatbot interface
│   └── recommend.html                 # Dropdown recommender
└── *.pkl                              # Serialized model artifacts
```

---

Feel free to explore, break it, and drop your feedback in the issues — all suggestions welcome.

---

<p align="center">
  Built by <a href="https://github.com/OwaisZafar01">Muhammad Owais Zafar</a>
</p>
