# 🎬 Cinematic AI — Intelligent Movie Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/LLaMA_3.1-8B_Instant-purple?style=for-the-badge&logo=meta&logoColor=white"/>
  <img src="https://img.shields.io/badge/BERT-all--MiniLM--L6--v2-orange?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/TMDB-API-01b4e4?style=for-the-badge&logo=themoviedatabase&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAG-Pipeline-green?style=for-the-badge"/>
</p>

<p align="center">
  A full-stack conversational AI movie recommender combining <strong>Content-Based Filtering</strong>, <strong>Collaborative Filtering</strong>, <strong>BERT Semantic Search</strong>, and <strong>LLM-powered response generation</strong> via a RAG (Retrieval-Augmented Generation) pipeline — all served through a Flask web application.
</p>

---

## 📌 Table of Contents
- [System Architecture](#-system-architecture)
- [RAG Pipeline Flow](#-rag-pipeline-flow)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CINEMATIC AI SYSTEM                             │
│                                                                         │
│  ┌──────────────┐     ┌──────────────────────────────────────────────┐  │
│  │   Frontend   │     │              Flask Backend                   │  │
│  │  (Jinja2 /   │────▶│                                              │  │
│  │   HTML/JS)   │     │  ┌────────────┐   ┌──────────────────────┐  │  │
│  └──────────────┘     │  │ /api/chat  │   │    /recommend        │  │  │
│                        │  │  (RAG)     │   │  (Content/Hybrid)    │  │  │
│                        │  └─────┬──────┘   └──────────────────────┘  │  │
│                        │        │                                      │  │
│                        │        ▼                                      │  │
│                        │  ┌───────────────────────────────────────┐   │  │
│                        │  │         RAG Pipeline                   │   │  │
│                        │  │  1. Context Builder                    │   │  │
│                        │  │  2. BERT Semantic Retriever (top-30)   │   │  │
│                        │  │  3. Groq LLaMA 3.1 Generator           │   │  │
│                        │  │  4. TMDB Poster Fetcher (parallel)     │   │  │
│                        │  └───────────────────────────────────────┘   │  │
│                        └──────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                      Model & Data Layer                          │    │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐   │    │
│  │  │  Content-Based  │  │  Collaborative   │  │ BERT Embeddings│  │    │
│  │  │  (CountVect +   │  │  Filtering (CF)  │  │ (384-dim,     │  │    │
│  │  │  Cosine Sim)    │  │  User–Item Matrix│  │  ~4800 movies)│  │    │
│  │  └─────────────────┘  └──────────────────┘  └───────────────┘   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Pipeline Flow

```
User Message
     │
     ▼
┌─────────────────────────┐
│  1. Context Builder      │  ◀── Conversation history (last 3 user turns)
│  Expands short queries   │      Short query (≤4 words)? → Prepend context
│  with conversation ctx   │      New intent? → Use message as-is
└────────────┬────────────┘
             │  Expanded Query
             ▼
┌─────────────────────────┐
│  2. BERT Semantic Search │  ◀── movie_embeddings.pkl (384-dim vectors)
│  all-MiniLM-L6-v2        │      Cosine similarity over ~4,800 movies
│  → Top-30 candidates     │      Returns: title, tags snippet, score
└────────────┬────────────┘
             │  Retrieved Context (top-30)
             ▼
┌─────────────────────────┐
│  3. Groq LLaMA 3.1-8B   │  ◀── System prompt: "Recommend ONLY from this list"
│  Instant LLM             │      History: last 5 turns (cleaned)
│  → JSON: reply + movies  │      Output: { "reply": "...", "movies": [...] }
└────────────┬────────────┘
             │  Selected Titles
             ▼
┌─────────────────────────┐
│  4. TMDB Poster Fetch    │  ◀── ThreadPoolExecutor (10 workers)
│  Parallel API calls      │      Fallback: placeholder image
│  → Poster URLs           │
└────────────┬────────────┘
             │
             ▼
        JSON Response
    { reply, movies[ {title, reason, poster} ] }
```

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🤖 **AI Chatbot** | Context-aware conversational movie recommendations via RAG |
| 🧠 **BERT Semantic Search** | Natural language understanding using `all-MiniLM-L6-v2` |
| 🦙 **LLaMA 3.1 via Groq** | Ultra-fast LLM inference for response generation |
| 🎯 **Hybrid Recommender** | Content-Based (60%) + Collaborative Filtering (40%) |
| 🔥 **Cold Start Handling** | Falls back to content-only if movie not in CF matrix |
| 🌐 **TMDB Integration** | Real-time poster fetching with parallel HTTP requests |
| 📊 **Popular Movies Feed** | Curated homepage with highly-rated TMDB movies |
| 💬 **Multi-turn Dialogue** | Maintains conversation context across multiple queries |
| ⚡ **Async Poster Fetching** | `ThreadPoolExecutor` for non-blocking TMDB API calls |

---

## 🛠 Tech Stack

### AI / ML
| Component | Technology |
|-----------|-----------|
| Semantic Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Similarity | `scikit-learn` cosine similarity |
| Content-Based Filtering | `CountVectorizer` + Cosine Similarity |
| Collaborative Filtering | User–Item Pivot Table + Cosine Similarity |
| NLP Tags | NLTK `PorterStemmer` |
| LLM Inference | Groq API — `llama-3.1-8b-instant` |

### Backend
| Component | Technology |
|-----------|-----------|
| Web Framework | Flask 3.x |
| Async I/O | `concurrent.futures.ThreadPoolExecutor` |
| Serialization | `pickle` |
| Environment Config | `python-dotenv` |

### External APIs
| API | Usage |
|-----|-------|
| TMDB API | Movie poster images, metadata |
| Groq API | LLaMA 3.1 chat completions |

---

## 📦 Dataset

The system is trained on two datasets:

**TMDB 5000 Movie Dataset**
- `tmdb_5000_movies.csv` — Movie metadata (genres, keywords, overview)
- `tmdb_5000_credits.csv` — Cast and crew data

**MovieLens Dataset** (for Collaborative Filtering)
- `movies.csv`, `ratings.csv`, `links.csv`
- Filtered: users with ≥50 ratings, movies with ≥20 ratings

### Preprocessing Pipeline

```
Raw Data
  │
  ├─ Merge movies + credits on title
  ├─ Extract: genres, keywords, top-3 cast, director
  ├─ Tokenize overview → word list
  ├─ Remove spaces from multi-word entities (e.g. "Science Fiction" → "ScienceFiction")
  ├─ Concatenate all fields into unified `tags` column
  ├─ Apply PorterStemmer to normalize vocabulary
  │
  ├─ Content Path:
  │     CountVectorizer (max 5000 features) → Cosine Similarity Matrix
  │
  ├─ BERT Path:
  │     SentenceTransformer.encode(tags) → 384-dim embeddings (batch_size=64)
  │
  └─ CF Path:
        MovieLens bridge via TMDB IDs → User-Item Pivot → Cosine Similarity
```

---

## 📁 Project Structure

```
cinematic-ai/
│
├── app.py                    # Flask app — routes and API endpoints
├── movie_recommendation_system.ipynb  # Full ML pipeline (training notebook)
│
├── templates/
│   ├── home.html             # Popular movies homepage
│   ├── chat.html             # AI chatbot interface
│   └── recommend.html        # Dropdown-based content/hybrid recommender
│
├── *.pkl                     # Serialized model artifacts:
│   ├── movies_df.pkl         # Cleaned movie dataframe
│   ├── similarity.pkl        # Content-based cosine similarity matrix
│   ├── movie_embeddings.pkl  # BERT semantic embeddings (4800 × 384)
│   ├── cf_similarity.pkl     # Collaborative filtering similarity matrix
│   └── popular_movies.pkl    # Top-rated movies for homepage
│
├── .env                      # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 1. Content-Based Filtering
Tags are constructed from overview + genres + keywords + top-3 cast + director, then stemmed and vectorized using `CountVectorizer(max_features=5000)`. Pairwise cosine similarity gives a `(4800 × 4800)` similarity matrix used for dropdown-based recommendations.

### 2. Collaborative Filtering (CF)
A user–item rating matrix is built from MovieLens data (filtered to active users and popular movies), bridged to TMDB titles via TMDB IDs. Cosine similarity across movie rows gives CF scores.

### 3. Hybrid Recommendation
```python
hybrid_score = 0.6 * content_score + 0.4 * cf_score
# Cold start: if movie not in CF matrix → 100% content-based
```

### 4. RAG Chatbot
```
User query → Context expansion → BERT retrieval (top-30) → LLaMA 3.1 selection → JSON response
```
The LLM is grounded to ONLY recommend from the BERT-retrieved candidates, preventing hallucination.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- TMDB API Key — [get one here](https://www.themoviedb.org/settings/api)
- Groq API Key — [get one here](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cinematic-ai.git
cd cinematic-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
flask
sentence-transformers
scikit-learn
numpy
pandas
nltk
groq
requests
python-dotenv
```

</details>

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Generate pickle files

Run the full notebook `movie_recommendation_system.ipynb` in Jupyter to generate all `.pkl` artifacts. This step requires the TMDB and MovieLens CSV files.

### 5. Run the application

```bash
python app.py
```

Visit `http://localhost:5000`

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
TMDB_API_KEY=your_tmdb_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📡 API Reference

### `POST /api/chat`

Send a message to the AI chatbot.

**Request Body:**
```json
{
  "message": "suggest a mind-bending sci-fi thriller",
  "history": [
    { "role": "user", "content": "I love Christopher Nolan movies" },
    { "role": "assistant", "content": "Here are some picks..." }
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your love for complex narratives...",
  "movies": [
    {
      "title": "Interstellar",
      "reason": "Mind-bending space epic with psychological depth",
      "poster": "https://image.tmdb.org/t/p/w500/..."
    }
  ]
}
```

### `POST /recommend`

Get content/hybrid recommendations for a specific movie title (form POST).

**Form Fields:**
- `movie` — exact movie title from the database

**Response:** Renders `recommend.html` with matched movie posters.

---

## 🗺 Roadmap

- [ ] User authentication and personalized history
- [ ] Streaming LLM responses for the chatbot
- [ ] Genre and mood filters on the chatbot UI
- [ ] Deployed demo on Render / Railway
- [ ] Integration tests for the RAG pipeline

---

## 🙏 Acknowledgements

- [TMDB](https://www.themoviedb.org/) for the movie database and poster API
- [Groq](https://groq.com/) for blazing-fast LLaMA inference
- [Hugging Face](https://huggingface.co/) for `sentence-transformers`
- [MovieLens](https://grouplens.org/datasets/movielens/) for the collaborative filtering dataset

---

<p align="center">Built with ❤️ using Flask · BERT · LLaMA 3.1 · TMDB API</p>
