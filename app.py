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
    with ThreadPoolExecutor(max_workers=10) as ex:
        return list(ex.map(fetch_poster, titles))

# ================= NEW: MOVIE DETAILS + TRAILER =================

def fetch_movie_details(title):
    """TMDB se movie ki full details aur YouTube trailer ID fetch karna"""
    try:
        # Step 1: Search movie to get TMDB ID
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"query": title, "api_key": TMDB_API_KEY}
        r = requests.get(search_url, params=params, timeout=5)
        results = r.json().get('results', [])
        if not results:
            return None

        movie = results[0]
        tmdb_id = movie['id']

        # Step 2: Fetch full details
        detail_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        detail_params = {"api_key": TMDB_API_KEY, "append_to_response": "credits"}
        d = requests.get(detail_url, params=detail_params, timeout=5).json()

        # Step 3: Fetch trailer from videos endpoint
        video_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos"
        v = requests.get(video_url, params={"api_key": TMDB_API_KEY}, timeout=5).json()

        trailer_key = None
        videos = v.get('results', [])
        # Priority: Official Trailer > Teaser > any YouTube video
        for vtype in ['Trailer', 'Teaser', 'Clip']:
            for vid in videos:
                if vid.get('site') == 'YouTube' and vid.get('type') == vtype:
                    trailer_key = vid['key']
                    break
            if trailer_key:
                break

        # Extract cast (top 5)
        cast = []
        for c in d.get('credits', {}).get('cast', [])[:5]:
            cast.append(c.get('name', ''))

        # Extract genres
        genres = [g['name'] for g in d.get('genres', [])]

        # Poster
        poster_path = d.get('poster_path')
        poster = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Poster"

        # Backdrop
        backdrop_path = d.get('backdrop_path')
        backdrop = f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else None

        return {
            "title":       d.get('title', title),
            "overview":    d.get('overview', 'No description available.'),
            "rating":      round(d.get('vote_average', 0), 1),
            "votes":       d.get('vote_count', 0),
            "release":     d.get('release_date', 'N/A')[:4] if d.get('release_date') else 'N/A',
            "runtime":     d.get('runtime', 0),
            "genres":      genres,
            "cast":        cast,
            "poster":      poster,
            "backdrop":    backdrop,
            "trailer_key": trailer_key,
            "trailer_url": f"https://www.youtube.com/watch?v={trailer_key}" if trailer_key else None,
        }

    except Exception as e:
        print(f"DEBUG: fetch_movie_details error -> {e}")
        return None


# ================= CORE AI LOGIC =================

def build_contextual_query(user_message, history):
    print(f"\n--- [DEBUG: Context Builder] ---")
    if not history:
        print("DEBUG: No history found.")
        return user_message
    past_user_msgs = [h['content'] for h in history if h['role'] == 'user'][-3:]
    context_str = " ".join(past_user_msgs)
    if len(user_message.split()) <= 4:
        full_query = f"{context_str} {user_message}"
        print(f"DEBUG: Expanded Query -> {full_query}")
    else:
        full_query = user_message
        print(f"DEBUG: New Intent Detected -> {full_query}")
    return full_query

def bert_retrieve(query, top_k=30):
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
    context_block = "\n".join([f"- {m['title']}: {m['tags']}" for m in retrieved_movies])
    system_prompt = f"""You are 'Movie Agent'. Recommend ONLY from this list:
    {context_block}

    Rules:
    1. If user says 'latest/more/before', use history to maintain genre/mood.
    2. Respond strictly in JSON format.
    3. format: {{"reply": "your text", "movies": [{{"title": "name", "reason": "why"}}]}}
    """
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
    expanded_query = build_contextual_query(user_msg, history)
    retrieved = bert_retrieve(expanded_query)
    result = ask_groq(user_msg, retrieved, history)
    recommended = result.get("movies", [])
    titles = [m["title"] for m in recommended if "title" in m]
    posters = fetch_posters_parallel(titles)
    final_movies = []
    for m, p in zip(recommended, posters):
        final_movies.append({
            "title":  m.get("title", "Unknown"),
            "reason": m.get("reason", "Great match for you"),
            "poster": p
        })
    print(f"DEBUG: Response Ready. Movies sent: {len(final_movies)}")
    return jsonify({
        "reply":  result.get("reply", "Here are your picks!"),
        "movies": final_movies
    })

# ================= NEW: MOVIE DETAILS API =================

@app.route('/api/movie-details', methods=['GET'])
def movie_details_api():
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400
    details = fetch_movie_details(title)
    if not details:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(details)

# ================= RECOMMEND ROUTE =================

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    all_movies = sorted(movies_df['title'].dropna().tolist())
    if request.method == 'POST':
        movie = request.form.get('movie', '').strip()
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
    
    