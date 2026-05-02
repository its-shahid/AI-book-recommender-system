import pickle
import streamlit as st
import numpy as np
import requests

st.set_page_config(page_title="Book Recommender System", layout="wide", page_icon="📚")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #fdf8f2;
    color: #2c1a0e;
}
h1, h2, h3 { font-family: 'Playfair Display', serif; }
.stApp { background-color: #fdf8f2; }

.hero {
    background: linear-gradient(135deg, #2c1a0e 0%, #5c3317 60%, #8b5e3c 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: #fdf8f2;
}
.hero h1 { color: #f5d9b0; font-size: 2.6rem; margin-bottom: 0.3rem; }
.hero p  { color: #d4b896; font-size: 1.05rem; margin: 0; }

.book-card {
    background: #fff8f0;
    border: 1px solid #e8d5c0;
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.book-card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(92,51,23,0.15); }
.book-card img { border-radius: 6px; width: 100%; max-height: 190px; object-fit: cover; }
.book-title { font-family: 'Playfair Display', serif; font-size: 0.82rem; color: #3d1f0a;
              margin-top: 0.5rem; font-weight: 600; line-height: 1.3; }

.ai-box {
    background: linear-gradient(135deg, #fff3e0, #fce8cc);
    border-left: 4px solid #c0692a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #3d1f0a;
    line-height: 1.6;
}
.ai-label { font-weight: 700; color: #c0692a; font-size: 0.78rem;
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }

div[data-testid="stSelectbox"] > div { border-color: #c0692a !important; border-radius: 8px; }
button[kind="primary"], .stButton > button {
    background: #5c3317 !important;
    color: #fdf8f2 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.55rem 2rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #8b5e3c !important; }

.stat-chip {
    display: inline-block;
    background: #5c3317;
    color: #f5d9b0;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    model        = pickle.load(open('model.pkl',        'rb'))
    book_names   = pickle.load(open('book_names.pkl',   'rb'))
    final_rating = pickle.load(open('final_rating.pkl', 'rb'))
    book_pivot   = pickle.load(open('book_pivot.pkl',   'rb'))

    rename_map = {
        'Book-Title':  'title',
        'Image-URL-L': 'image_url',
        'User-ID':     'user_id',
        'Book-Rating': 'rating',
    }
    cols_to_rename = {k: v for k, v in rename_map.items() if k in final_rating.columns}
    if cols_to_rename:
        final_rating.rename(columns=cols_to_rename, inplace=True)

    return model, book_names, final_rating, book_pivot

model, book_names, final_rating, book_pivot = load_data()


def fetch_poster(suggestions):
    poster_urls = []
    for book_id in suggestions[0]:
        book_name = book_pivot.index[book_id]
        matches   = final_rating[final_rating['title'] == book_name]
        url = matches.iloc[0]['image_url'] if not matches.empty else "https://via.placeholder.com/150x200?text=No+Cover"
        poster_urls.append(url)
    return poster_urls


def recommend_book(book_name):
    if book_name not in book_pivot.index:
        return [], [], "OUT_OF_DOMAIN"

    book_id = np.where(book_pivot.index == book_name)[0][0]
    _, suggestions = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=6
    )
    poster_urls = fetch_poster(suggestions)
    books_list  = [book_pivot.index[i] for i in suggestions[0]]
    return books_list, poster_urls, "OK"


def get_smart_explanation(selected_book: str, recommended_books: list[str]) -> str:
    rec_list = ", ".join(recommended_books[:5])
    prompt = (
        f"A reader enjoyed the book '{selected_book}'. "
        f"A collaborative-filtering model recommended these books: {rec_list}. "
        "In 3–4 sentences, explain what themes, style, or audience these books "
        "share with the original. Be warm, specific, and helpful. "
        "Do NOT recommend books outside this list or answer unrelated questions."
    )
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": (
                    "You are a book recommendation assistant. "
                    "You ONLY discuss books and reading. "
                    "If asked anything unrelated to books or literature, "
                    "politely decline and redirect to books."
                ),
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        data = response.json()
        return data["content"][0]["text"].strip()
    except Exception as e:
        return f"_(Explanation engine unavailable: {e})_"


st.markdown("""
<div class="hero">
  <h1>📚 Book Recommender System</h1>
  <p>Discover your next favourite read — powered by collaborative filtering & intelligent analysis</p>
  <p style="margin-top:0.8rem; font-size:0.85rem; color:#f5d9b0; opacity:0.85;">
    🎓 Powered by <strong>Lovely Professional University</strong>
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<span class="stat-chip">📖 {len(book_names):,} Books</span>'
    f'<span class="stat-chip">🤖 Smart Recommendations</span>'
    f'<span class="stat-chip">⭐ Collaborative Filtering</span>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

col_sel, col_btn = st.columns([4, 1])
with col_sel:
    selected_book = st.selectbox(
        "🔍 Type or select a book from the catalogue",
        book_names,
        help="Only books present in our catalogue are supported."
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    show = st.button("Recommend →", use_container_width=True)


if show:
    if selected_book not in list(book_pivot.index):
        st.error(
            "❌ **Out of domain:** This book is not in our catalogue. "
            "Please select a book from the dropdown list."
        )
        st.stop()

    with st.spinner("Finding the best matches for you…"):
        books_list, poster_urls, status = recommend_book(selected_book)

    if status == "OUT_OF_DOMAIN" or not books_list:
        st.error("❌ Could not find recommendations for this title. Please try another book.")
        st.stop()

    display_books  = books_list[1:6]
    display_covers = poster_urls[1:6]

    st.markdown("---")
    st.subheader(f"📖 Because you liked: *{selected_book}*")

    cols = st.columns(5)
    for i, (title, cover) in enumerate(zip(display_books, display_covers)):
        with cols[i]:
            st.markdown(f"""
            <div class="book-card">
              <img src="{cover}" alt="{title}" onerror="this.src='https://via.placeholder.com/150x200?text=No+Cover'"/>
              <div class="book-title">{title}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("🔍 Analysing why these books match…"):
        explanation = get_smart_explanation(selected_book, display_books)

    st.markdown(f"""
    <div class="ai-box">
      <div class="ai-label">💡 Why these books match</div>
      {explanation}
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "ℹ️ **Domain notice:** This system only recommends books. "
        "Queries outside the book domain are not supported.",
        icon="📚"
    )

    st.markdown("""
    <div style='text-align:center; margin-top:3rem; color:#8b5e3c; font-size:0.8rem;'>
        🎓 <strong>Lovely Professional University</strong> &nbsp;|&nbsp; INT428 Skill Based Assignment &nbsp;|&nbsp; Book Recommender System
    </div>
    """, unsafe_allow_html=True)