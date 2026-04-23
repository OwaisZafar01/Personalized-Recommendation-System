import os
import pickle
import requests
import numpy as np
import json, re
from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
# ================= CONFIGURATION & KEYS =================
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# ================= LOAD MODELS & DATA =================
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    try:
        data = {
            "movies": pickle.load(open('movies_df.pkl', 'rb')),
            "similarity": pickle.load(open('similarity.pkl', 'rb')),
            "cf_sim": pickle.load(open('cf_similarity.pkl', 'rb')),
            "embeddings": pickle.load(open('movie_embeddings.pkl', 'rb')),
            "popular": pickle.load(open('popular_movies.pkl', 'rb'))
        }
        return data
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load pickle files: {e}")
        return None

db = load_data()
movies_df = db["movies"]
similarity = db["similarity"]
cf_sim_df = db["cf_sim"]
movie_embeddings = db["embeddings"]
popular_df = db["popular"]

# ================= HELPER FUNCTIONS =================

def fetch_poster(movie_title):
    """TMDB API se poster link fetch karna"""
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"query": movie_title, "api_key": TMDB_API_KEY}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data.get('results') and data['results'][0].get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['results'][0]['poster_path']
    except Exception:
        pass
    return "https://via.placeholder.com/300x450?text=No+Poster"

def fetch_posters_parallel(titles):
    """Multiple posters ko ek saath fetch karna for speed"""
    with ThreadPoolExecutor(max_workers=10) as ex:
        return list(ex.map(fetch_poster, titles))

# ================= CORE AI LOGIC =================

def build_contextual_query(user_message, history):
    """History aur current message ko merge karke context banana"""
    print(f"\n--- [DEBUG: Context Builder] ---")
    if not history:
        print("DEBUG: No history found.")
        return user_message

    # Pichle 3 messages ka user context
    past_user_msgs = [h['content'] for h in history if h['role'] == 'user'][-3:]
    context_str = " ".join(past_user_msgs)
    
    # Logic: Agar user chota sentence bole (e.g. 'latest'), toh history jodo
    if len(user_message.split()) <= 4:
        full_query = f"{context_str} {user_message}"
        print(f"DEBUG: Expanded Query -> {full_query}")
    else:
        full_query = user_message
        print(f"DEBUG: New Intent Detected -> {full_query}")
    
    return full_query

def bert_retrieve(query, top_k=30):
    """BERT Semantic Search results database se nikalna"""
    q_emb = bert_model.encode([query])
    scores = cosine_similarity(q_emb, movie_embeddings).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    
    print(f"DEBUG: BERT found {top_k} candidate movies.")
    return [{
        'title': movies_df.iloc[i]['title'],
        'tags' : movies_df.iloc[i]['tags'][:200],
        'score': float(scores[i])
    } for i in top_idx]

def ask_groq(user_message, retrieved_movies, history):
    """LLM se natural response aur selection karwana"""
    context_block = "\n".join([f"- {m['title']}: {m['tags']}" for m in retrieved_movies])
    
    system_prompt = f"""You are 'Movie Agent'. Recommend ONLY from this list:
    {context_block}

    Rules:
    1. If user says 'latest/more/before', use history to maintain genre/mood.
    2. Respond strictly in JSON format.
    3. format: {{"reply": "your text", "movies": [{{"title": "name", "reason": "why"}}]}}
    """

    # History cleaning (Assistant ke JSON messages se text nikalna)
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-5:]:
        content = h['content']
        if h['role'] == 'assistant' and isinstance(content, str) and '{' in content:
            try: content = json.loads(content).get('reply', 'Picks for you.')
            except: pass
        messages.append({"role": h['role'], "content": str(content)})
    
    messages.append({"role": "user", "content": user_message})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        
        # Robust JSON Parsing
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
        return {"reply": raw, "movies": []}
    except Exception as e:
        print(f"DEBUG: Groq Logic Error -> {e}")
        return {"reply": "I'm having a technical glitch. Try again?", "movies": []}

# ================= ROUTES =================

@app.route('/')
def home():
    titles = popular_df['title'].tolist()
    ratings = popular_df['vote_average'].tolist()
    posters = fetch_posters_parallel(titles)
    movies = [{"title": t, "poster": p, "rating": round(r, 1)} for t, p, r in zip(titles, posters, ratings)]
    return render_template("home.html", movies=movies)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # 1. Expand Query with Context
    expanded_query = build_contextual_query(user_msg, history)

    # 2. Semantic Search
    retrieved = bert_retrieve(expanded_query)

    # 3. AI Selection
    result = ask_groq(user_msg, retrieved, history)

    # 4. Final Formatting & Posters
    recommended = result.get("movies", [])
    titles = [m["title"] for m in recommended if "title" in m]
    posters = fetch_posters_parallel(titles)

    final_movies = []
    for m, p in zip(recommended, posters):
        final_movies.append({
            "title": m.get("title", "Unknown"),
            "reason": m.get("reason", "Great match for you"),
            "poster": p
        })

    print(f"DEBUG: Response Ready. Movies sent: {len(final_movies)}")
    return jsonify({
        "reply": result.get("reply", "Here are your picks!"),
        "movies": final_movies
    })

# Collaborative / Hybrid Route (Dropdown Search)
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    all_movies = sorted(movies_df['title'].dropna().tolist())
    if request.method == 'POST':
        movie = request.form.get('movie', '').strip()
        # Simple retrieval from content similarity for this page
        if movie in movies_df['title'].values:
            idx = movies_df[movies_df['title'] == movie].index[0]
            scores = list(enumerate(similarity[idx]))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:7]
            titles = [movies_df.iloc[i[0]]['title'] for i in sorted_scores]
            posters = fetch_posters_parallel(titles)
            return render_template('recommend.html', movies=list(zip(titles, posters)), all_movies=all_movies)
    return render_template('recommend.html', all_movies=all_movies)

if __name__ == '__main__':
    app.run(debug=True, port=5000)