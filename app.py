import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PageTurn · Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  —  Modern Aesthetic / Editorial
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400;1,600&family=Inter:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --cream:        #faf7f2;
    --warm-white:   #f5f0e8;
    --parchment:    #ede8de;
    --ink:          #1a1410;
    --ink-light:    #3d342a;
    --ink-muted:    #5a4e42;
    --ink-faint:    #8a7e72;
    --terracotta:   #c2714f;
    --terracotta-l: #d4896a;
    --sage:         #6b8f71;
    --dusty-blue:   #5b7fa6;
    --gold:         #c9a84c;
    --border:       rgba(26,20,16,.15);
    --border-warm:  rgba(194,113,79,.30);
    --shadow-sm:    0 1px 3px rgba(26,20,16,.12);
    --shadow-md:    0 4px 16px rgba(26,20,16,.15);
}

/* ── Font fallbacks so text is always visible ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
    color: var(--ink) !important;
}

/* ── FIX: Force ALL main-area text to dark ink (prevents Streamlit dark-mode bleed) ── */
.main .block-container,
.main .block-container * {
    color: #1a1410;
}

/* ── FIX: Widget labels — Streamlit renders these as near-white in dark mode ── */
/* Covers: text inputs, number inputs, selectboxes, sliders, all label elements */
.main label,
.main label p,
.main label span,
.main .stTextInput label,
.main .stNumberInput label,
.main .stSelectbox label,
.main .stSlider label,
.main .stSlider [data-testid="stWidgetLabel"],
.main .stSlider [data-testid="stWidgetLabel"] p,
.main .stTextInput [data-testid="stWidgetLabel"],
.main .stTextInput [data-testid="stWidgetLabel"] p,
.main .stNumberInput [data-testid="stWidgetLabel"],
.main .stNumberInput [data-testid="stWidgetLabel"] p,
.main .stSelectbox [data-testid="stWidgetLabel"],
.main .stSelectbox [data-testid="stWidgetLabel"] p,
[data-testid="stExpander"] label,
[data-testid="stExpander"] label p,
[data-testid="stExpander"] label span,
[data-testid="stExpander"] [data-testid="stWidgetLabel"],
[data-testid="stExpander"] [data-testid="stWidgetLabel"] p,
[data-testid="stExpander"] [data-testid="stWidgetLabel"] span {
    color: #1a1410 !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
}

/* Slider min/max tick values */
.main [data-testid="stSlider"] [data-testid="stTickBarMin"],
.main [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stExpander"] [data-testid="stTickBarMin"],
[data-testid="stExpander"] [data-testid="stTickBarMax"] {
    color: #5a4e42 !important;
    font-size: .68rem !important;
}

/* Current slider value bubble */
.main [data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stExpander"] [data-testid="stThumbValue"] {
    color: #1a1410 !important;
    background: rgba(245,240,232,.95) !important;
    border: 1px solid rgba(194,113,79,.3) !important;
    font-weight: 600 !important;
}

/* Number input stepper buttons */
.main [data-testid="stNumberInput"] button,
[data-testid="stExpander"] [data-testid="stNumberInput"] button {
    color: #1a1410 !important;
    background: rgba(237,232,222,.8) !important;
    border: 1px solid rgba(26,20,16,.15) !important;
}

/* Selectbox selected value text */
.main [data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"],
.main [data-baseweb="select"] > div > div,
[data-testid="stExpander"] [data-baseweb="select"] > div > div,
[data-testid="stExpander"] [data-baseweb="select"] span {
    color: #1a1410 !important;
    background: #ffffff !important;
}

/* ── FIX: Title h1 — Streamlit overrides heading color to near-white ── */
.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.pt-title {
    color: #1a1410 !important;
}

/* ── FIX: Expander header — dark background bleed ── */
[data-testid="stExpander"],
[data-testid="stExpander"] > details,
[data-testid="stExpander"] > details > summary {
    background: rgba(245,240,232,.96) !important;
    color: #1a1410 !important;
    border: 1px solid rgba(26,20,16,.14) !important;
}
[data-testid="stExpander"] > details > summary:hover {
    background: rgba(237,232,222,.98) !important;
}
[data-testid="stExpander"] > details > summary p,
[data-testid="stExpander"] > details > summary span,
[data-testid="stExpander"] > details > summary svg {
    color: #1a1410 !important;
    fill: #1a1410 !important;
}
/* Expander body */
[data-testid="stExpander"] > details > div {
    background: rgba(250,247,242,.97) !important;
}

/* ── FIX: Main area inputs (inside expanders) — dark bg bleed ── */
.main [data-testid="stTextInput"] input,
.main [data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #1a1410 !important;
    border: 1px solid rgba(26,20,16,.2) !important;
    border-bottom: 2px solid #c2714f !important;
    border-radius: 3px !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: .9rem !important;
}
.main [data-testid="stTextInput"] input::placeholder {
    color: #8a7e72 !important;
}

/* ── FIX: Main area selectbox ── */
.main [data-baseweb="select"] > div,
.main [data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    color: #1a1410 !important;
    border: 1px solid rgba(26,20,16,.2) !important;
}
.main [data-baseweb="select"] span,
.main [data-baseweb="select"] div {
    color: #1a1410 !important;
}
/* Dropdown list items */
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li {
    background: #faf7f2 !important;
    color: #1a1410 !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: rgba(194,113,79,.1) !important;
}

/* ── FIX: Main area slider labels and values ── */
.main [data-testid="stSlider"] label,
.main [data-testid="stSlider"] p,
.main [data-testid="stSlider"] span {
    color: #1a1410 !important;
    font-family: 'DM Mono', 'Courier New', monospace !important;
    font-size: .72rem !important;
}
/* Slider track fill */
.main [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"],
.main [data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {
    background: #c2714f !important;
}
.main [data-testid="stSlider"] [role="slider"] {
    background: #c9a84c !important;
    border-color: #c9a84c !important;
}

/* ── FIX: Streamlit markdown text in main area ── */
.main .stMarkdown p,
.main .stMarkdown span,
.main .stMarkdown div {
    color: #1a1410 !important;
}

/* ── Background: image visible at 45% opacity, lighter overlay ── */
.stApp {
    background-color: #f0ebe0 !important;
    background-image:
        linear-gradient(rgba(245,240,232,.72), rgba(240,235,224,.75)),
        url('https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=1800&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1410 !important;
    border-right: 1px solid rgba(201,168,76,.25) !important;
}
[data-testid="stSidebar"] * {
    color: #c8bfb5 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'DM Mono', 'Courier New', monospace !important;
    font-size: .75rem !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    color: #a09890 !important;
}
/* Sidebar input text visible — scoped so .main rules don't override */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    color: #faf7f2 !important;
    background: rgba(255,255,255,.08) !important;
    border: none !important;
    border-bottom: 2px solid #c9a84c !important;
    border-radius: 0 !important;
}
/* Sidebar slider track */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #c9a84c !important;
}

/* ── Header ── */
.pt-header { text-align: center; padding: 3rem 1rem 2rem; }
.pt-eyebrow {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .72rem; letter-spacing: .3em;
    text-transform: uppercase; color: #c2714f;
    margin-bottom: .8rem; font-weight: 600;
}
.pt-title {
    font-family: 'Playfair Display', Georgia, 'Times New Roman', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 900; color: #1a1410;
    line-height: 1.05; margin: 0 0 .4rem;
    text-shadow: 0 2px 8px rgba(26,20,16,.12);
}
.pt-title em { color: #c2714f; font-style: italic; }
.pt-sub {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem; color: #4a3e32;
    font-style: italic; margin-top: .4rem;
    transition: opacity .5s ease;
    text-shadow: 0 1px 3px rgba(255,255,255,.5);
}
.pt-divider {
    width: 60px; height: 2px;
    background: #c2714f;
    margin: 1.4rem auto; border-radius: 2px;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 1rem; margin: 1.5rem 0;
}
.metric-card {
    background: rgba(245,240,232,.92);
    border: 1px solid rgba(26,20,16,.15);
    border-top: 3px solid #c2714f;
    padding: 1.4rem 1.2rem; text-align: center;
    border-radius: 4px; box-shadow: var(--shadow-sm);
    transition: box-shadow .2s, transform .2s;
    backdrop-filter: blur(4px);
}
.metric-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.metric-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2rem; font-weight: 700; color: #1a1410; line-height: 1;
}
.metric-label {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .68rem; letter-spacing: .18em;
    color: #5a4e42; text-transform: uppercase; margin-top: .5rem;
    font-weight: 600;
}

/* ── Book cards ── */
.book-card {
    background: rgba(245,240,232,.94);
    border: 1px solid rgba(26,20,16,.14);
    border-left: 4px solid #c2714f;
    padding: 1.2rem 1.4rem; margin: .65rem 0;
    display: flex; align-items: flex-start; gap: 1.2rem;
    border-radius: 0 4px 4px 0; box-shadow: var(--shadow-sm);
    transition: all .22s ease; position: relative;
    backdrop-filter: blur(4px);
}
.book-card:hover {
    box-shadow: var(--shadow-md);
    border-left-color: #c9a84c;
    transform: translateX(3px);
    background: rgba(250,247,242,.97);
}
.book-rank {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.5rem; font-weight: 700;
    color: #8a7e72; min-width: 2.2rem; line-height: 1; margin-top: .1rem;
}
.book-cover { border-radius: 3px; min-width: 60px; box-shadow: 3px 3px 12px rgba(26,20,16,.2); }
.book-info { flex: 1; }
.book-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem; font-weight: 700; color: #1a1410;
    line-height: 1.35; margin-bottom: .3rem;
}
.book-author {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-size: .82rem; color: #c2714f; font-weight: 600;
}
.book-publisher {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .68rem; color: #7a6e62; margin-top: .2rem;
}
.book-score {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .72rem; color: #3d342a; text-align: right; min-width: 5rem;
}
.score-bar {
    height: 3px; background: linear-gradient(90deg, #c2714f, #c9a84c);
    border-radius: 2px; margin-top: .5rem;
}

/* ── Section badge ── */
.section-badge {
    display: inline-block;
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .65rem; font-weight: 600; letter-spacing: .2em;
    text-transform: uppercase; padding: .35rem 1rem;
    background: rgba(245,240,232,.92); border: 1px solid rgba(194,113,79,.35);
    color: #c2714f; border-radius: 3px; margin-bottom: 1rem;
}

/* Sidebar form inputs — handled by scoped sidebar rules above */

/* ── Button ── */
.stButton > button {
    background: #c2714f !important;
    border: none !important;
    color: #faf7f2 !important;
    font-family: 'DM Mono', 'Courier New', monospace !important;
    font-size: .75rem !important;
    font-weight: 600 !important;
    letter-spacing: .18em !important;
    padding: .8rem 2rem !important;
    border-radius: 3px !important;
    width: 100% !important;
    transition: all .2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #d4896a !important;
    box-shadow: 0 4px 16px rgba(194,113,79,.40) !important;
    transform: translateY(-1px) !important;
}

/* ── Slider thumb ── */
[data-testid="stSlider"] [role="slider"] { background: #c9a84c !important; }

/* ── Expander (handled by specific selectors above) ── */

/* ── Spinner ── */
.stSpinner > div { color: #c2714f !important; }

/* ── Alert ── */
[data-testid="stAlert"] {
    background: rgba(194,113,79,.08) !important;
    border: 1px solid rgba(194,113,79,.35) !important;
    border-radius: 4px !important;
    color: #1a1410 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Log box ── */
.log-line {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .73rem;
    color: #6b8f71; padding: .2rem 0;
    border-bottom: 1px solid rgba(107,143,113,.12);
    letter-spacing: .04em;
}
.log-box {
    background: #1a1410;
    border: 1px solid rgba(107,143,113,.25);
    border-left: 3px solid #6b8f71;
    padding: 1rem 1.2rem; max-height: 200px;
    overflow-y: auto; border-radius: 0 4px 4px 0;
}

/* ── User chip ── */
.user-chip {
    display: inline-flex; align-items: center; gap: .7rem;
    background: rgba(245,240,232,.95);
    border: 1px solid rgba(194,113,79,.35);
    padding: .6rem 1.2rem;
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .75rem; color: #3d342a;
    margin-bottom: 1.4rem; border-radius: 3px;
    box-shadow: var(--shadow-sm); letter-spacing: .05em;
    font-weight: 500;
}
.user-dot {
    width: 8px; height: 8px; background: #6b8f71;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .5; transform: scale(.8); }
}

/* ── Tags ── */
.tag {
    display: inline-block;
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .62rem; letter-spacing: .1em; padding: .25rem .7rem;
    border-radius: 3px; margin-right: .4rem; margin-top: .3rem;
    text-transform: uppercase; font-weight: 600;
}
.tag-item   { background: rgba(91,127,166,.12);  color: #3a6090; border: 1px solid rgba(91,127,166,.35); }
.tag-user   { background: rgba(194,113,79,.12);  color: #a0502d; border: 1px solid rgba(194,113,79,.35); }
.tag-hybrid { background: rgba(201,168,76,.12);  color: #9a7e2a; border: 1px solid rgba(201,168,76,.35); }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3.5rem 1rem;
    color: #5a4e42;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1rem; font-style: italic;
    border: 1px dashed rgba(194,113,79,.4);
    border-radius: 4px;
    background: rgba(245,240,232,.90);
}

/* ── Summary bar ── */
.summary-bar {
    display: flex; gap: 2rem; padding: .7rem 1rem; margin: .5rem 0;
    background: rgba(245,240,232,.92);
    border: 1px solid rgba(26,20,16,.12);
    border-left: 3px solid #c2714f;
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: .68rem; letter-spacing: .06em;
    border-radius: 0 4px 4px 0; flex-wrap: wrap;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING  (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df1 = pd.read_csv("Users.csv", encoding='latin1')
    df1['Age'] = pd.to_numeric(df1['Age'], errors='coerce')
    valid_age = (df1['Age'] >= 5) & (df1['Age'] <= 90)
    df1.loc[~valid_age, 'Age'] = None
    df1['Age'].fillna(df1['Age'].median(), inplace=True)
    df1['Age'] = df1['Age'].fillna(0)
    df1['Age'] = df1['Age'].astype(int)
    df1['Country'] = df1['Location'].apply(lambda x: x.split(',')[-1].strip())

    df2 = pd.read_csv('Ratings.csv', encoding='latin1')
    df2 = df2[df2['Book-Rating'] != 0]
    user_counts = df2['User-ID'].value_counts()
    active_users = user_counts[user_counts >= 20].index
    df2 = df2[df2['User-ID'].isin(active_users)]
    book_counts = df2['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= 5].index
    df2 = df2[df2['ISBN'].isin(popular_books)]

    df3 = pd.read_csv("Books.csv", encoding='latin1')
    df3['Year-Of-Publication'] = pd.to_numeric(df3['Year-Of-Publication'], errors='coerce')
    df3['Book-Author'].fillna('Unknown', inplace=True)
    df3['Publisher'].fillna('Unknown', inplace=True)
    mode_year = df3['Year-Of-Publication'].mode()[0]
    df3['Year-Of-Publication'] = df3['Year-Of-Publication'].fillna(mode_year)

    df4 = df1.merge(df2, on='User-ID')
    df = df4.merge(df3, on='ISBN')
    df = df[['User-ID','Country','Book-Title','ISBN','Book-Author','Publisher','Image-URL-M','Book-Rating']]
    return df


@st.cache_resource(show_spinner=False)
def build_matrices(_df):
    pivot_item = _df.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
    pivot_item = pivot_item.astype('float32')
    pivot_item_norm = pivot_item.subtract(pivot_item.mean(axis=1), axis=0).fillna(0)
    item_sim = cosine_similarity(pivot_item_norm)
    item_sim_df = pd.DataFrame(item_sim, index=pivot_item.index, columns=pivot_item.index)

    pivot_user = _df.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)
    pivot_user_norm = pivot_user.subtract(pivot_user.mean(axis=1), axis=0).fillna(0)
    user_sim = cosine_similarity(pivot_user_norm)
    user_sim_df = pd.DataFrame(user_sim, index=pivot_user.index, columns=pivot_user.index)

    return item_sim_df, user_sim_df


# ─────────────────────────────────────────────
#  RECOMMENDATION LOGIC
# ─────────────────────────────────────────────
def recommend_item_based(user_id, df, sim_df, n=20):
    user_data = df[df['User-ID'] == user_id]
    if user_data.empty:
        return []
    user_books = user_data['Book-Title'].tolist()
    scores = {}
    for book in user_books:
        if book in sim_df.index:
            similar = sim_df[book].sort_values(ascending=False)[1:11]
            rating = user_data[user_data['Book-Title'] == book]['Book-Rating'].mean()
            for sim_book, sim_score in similar.items():
                scores[sim_book] = scores.get(sim_book, 0) + sim_score * rating
    for book in user_books:
        scores.pop(book, None)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def recommend_user_based(user_id, df, sim_df, n=20):
    if user_id not in sim_df.index:
        return []
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:11]
    user_books = df[df['User-ID'] == user_id]['Book-Title'].tolist()
    scores = {}
    for sim_user, sim_score in similar_users.items():
        for _, row in df[df['User-ID'] == sim_user].iterrows():
            book = row['Book-Title']
            if book not in user_books:
                scores[book] = scores.get(book, 0) + sim_score * row['Book-Rating']
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def hybrid(item_recs, user_recs, iw=0.7, uw=0.3, n=10):
    scores = {}
    for book, score in item_recs:
        scores[book] = scores.get(book, 0) + score * iw
    for book, score in user_recs:
        scores[book] = scores.get(book, 0) + score * uw
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def get_book_details(recommendations, df):
    result = []
    for book, score in recommendations:
        rows = df[df['Book-Title'] == book]
        if rows.empty:
            continue
        temp = rows.iloc[0]
        result.append({
            "Book-Title": book,
            "Author":     temp['Book-Author'],
            "Publisher":  temp['Publisher'],
            "Image":      temp['Image-URL-M'],
            "Score":      round(float(score), 3),
        })
    return result


def recommend_for_user(user_id, df, item_sim_df, user_sim_df, item_weight, user_weight, n=10):
    item_recs = recommend_item_based(user_id, df, item_sim_df, n=20)
    user_recs = recommend_user_based(user_id, df, user_sim_df, n=20)
    final = hybrid(item_recs, user_recs, iw=item_weight, uw=user_weight, n=n)
    return get_book_details(final, df)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.6rem 0 1rem;">
        <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.4rem;font-weight:900;color:#faf7f2;">
            Page<em style="color:#c9a84c;font-style:italic;">Turn</em>
        </div>
        <div style="font-family:'DM Mono','Courier New',monospace;font-size:.58rem;color:#a09890;
                    letter-spacing:.22em;margin-top:.35rem;text-transform:uppercase;font-weight:500;">
            Book Recommendation Engine
        </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,.45),transparent);
                margin:.5rem 0 1.4rem;"></div>
    <div style="font-family:'DM Mono','Courier New',monospace;font-size:.6rem;letter-spacing:.2em;
                color:#a09890;text-transform:uppercase;margin-bottom:.6rem;font-weight:500;">
        — Reader Settings —
    </div>
    """, unsafe_allow_html=True)

    user_id_input = st.number_input("User ID", min_value=1, value=276725, step=1)
    n_recs = st.slider("RECOMMENDATION COUNT", min_value=3, max_value=20, value=10)

    st.markdown("""
    <div style="font-family:'DM Mono','Courier New',monospace;font-size:.6rem;letter-spacing:.2em;
                color:#a09890;text-transform:uppercase;margin:1rem 0 .4rem;font-weight:500;">
        — Engine Weights —
    </div>""", unsafe_allow_html=True)

    item_weight = st.slider("ITEM-BASED WEIGHT", 0.0, 1.0, 0.7, 0.05)
    user_weight = round(1.0 - item_weight, 2)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-family:'DM Mono','Courier New',monospace;
                font-size:.68rem;margin-top:.3rem;font-weight:600;">
        <span style="color:#c9a84c;">Item  {item_weight:.0%}</span>
        <span style="color:#d4896a;">User  {user_weight:.0%}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("✦  Find My Books")

    st.markdown("""
    <div style="margin-top:2rem;">
        <div style="font-family:'DM Mono','Courier New',monospace;font-size:.56rem;color:#5a4e42;
                    letter-spacing:.15em;text-transform:uppercase;margin-bottom:.6rem;font-weight:500;">
            — Reading Worlds —
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;">
            <img src="https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=200&q=70"
                 style="width:100%;height:72px;object-fit:cover;opacity:.75;border-radius:2px;
                        border:1px solid rgba(201,168,76,.25);">
            <img src="https://images.unsplash.com/photo-1512820790803-83ca734da794?w=200&q=70"
                 style="width:100%;height:72px;object-fit:cover;opacity:.75;border-radius:2px;
                        border:1px solid rgba(194,113,79,.25);">
            <img src="https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?w=200&q=70"
                 style="width:100%;height:72px;object-fit:cover;opacity:.75;border-radius:2px;
                        border:1px solid rgba(107,143,113,.25);">
            <img src="https://images.unsplash.com/photo-1507842217343-583bb7270b66?w=200&q=70"
                 style="width:100%;height:72px;object-fit:cover;opacity:.75;border-radius:2px;
                        border:1px solid rgba(91,127,166,.25);">
        </div>
        <div style="font-family:'Playfair Display',Georgia,serif;font-size:.75rem;color:#a09890;
                    margin-top:.5rem;text-align:center;font-style:italic;">
            Infinite stories await.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div class="pt-header">
    <div class="pt-eyebrow">AI · Collaborative Filtering · Hybrid Engine</div>
    <h1 class="pt-title">Page<em>Turn</em></h1>
    <p class="pt-sub" id="rotating-quote">"So many books, so little time — let the machine decide."</p>
    <div class="pt-divider"></div>
</div>
<script>
const _quotes = [
    "\u201cSo many books, so little time \u2014 let the machine decide.\u201d",
    "\u201cA human picks one book. We pick ten. You\u2019re welcome.\u201d",
    "\u201cReading lists curated by AI \u2014 your TBR pile just got a PhD.\u201d",
    "\u201cYes, we judged you by your reading history. No regrets.\u201d",
    "\u201cWe know what you\u2019ll read next before you do. Spooky? Just smart.\u201d",
    "\u201cYour next favourite book was chosen by math. Glamorous, isn\u2019t it?\u201d",
    "\u201cBecause asking a librarian is so last century.\u201d",
    "\u201cAlgorithmically better taste than your book club. Proven.\u201d",
];
let _qi = 0;
function _rotateQuote() {
    const el = document.getElementById('rotating-quote');
    if (!el) return;
    el.style.opacity = '0';
    setTimeout(() => {
        _qi = (_qi + 1) % _quotes.length;
        el.innerHTML = _quotes[_qi];
        el.style.opacity = '1';
    }, 500);
}
setInterval(_rotateQuote, 4000);
</script>
""", unsafe_allow_html=True)

# ── Load data ──
with st.spinner("Loading library…"):
    try:
        df = load_data()
        item_sim_df, user_sim_df = build_matrices(df)
        data_loaded = True
    except FileNotFoundError as e:
        data_loaded = False
        missing = str(e)

if not data_loaded:
    st.markdown(f"""
    <div style="text-align:center;padding:3rem;background:rgba(245,240,232,.95);
                border:1px dashed rgba(194,113,79,.4);border-radius:4px;">
        <div style="font-family:'Playfair Display',Georgia,serif;color:#c2714f;font-size:1.3rem;font-weight:700;">
            Data Files Not Found
        </div>
        <div style="font-family:'Inter','Segoe UI',sans-serif;color:#5a4e42;font-size:.88rem;margin-top:.8rem;">
            Place <code>Users.csv</code>, <code>Ratings.csv</code>, <code>Books.csv</code>
            in the same directory as this script.
        </div>
        <div style="font-family:'DM Mono','Courier New',monospace;color:#c2714f;font-size:.7rem;margin-top:.5rem;">{missing}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Stats row ──
total_users   = df['User-ID'].nunique()
total_books   = df['Book-Title'].nunique()
total_ratings = len(df)

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{total_users:,}</div>
        <div class="metric-label">Active Readers</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_books:,}</div>
        <div class="metric-label">Book Titles</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_ratings:,}</div>
        <div class="metric-label">Total Ratings</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── The User Storyboard ──
with st.expander("🔍  The User Storyboard  —  Full dataset user breakdown", expanded=False):

    st.markdown("""
    <div style="font-family:'Playfair Display',Georgia,serif;font-size:1rem;font-weight:700;
                color:#1a1410;margin-bottom:.3rem;">
        Active Reader Registry
        <span style="font-family:'DM Mono','Courier New',monospace;font-weight:500;font-size:.65rem;
                     color:#7a6e62;letter-spacing:.05em;margin-left:.8rem;">
            Only readers who rated ≥ 20 books appear here
        </span>
    </div>
    """, unsafe_allow_html=True)

    user_analysis = (
        df.groupby('User-ID')
        .agg(
            Books_Read=('Book-Title',   'nunique'),
            Total_Ratings=('Book-Rating','count'),
            Avg_Rating=('Book-Rating',  'mean'),
            Country=('Country',         'first'),
        )
        .reset_index()
    )
    user_analysis['Avg_Rating'] = user_analysis['Avg_Rating'].round(2)
    user_analysis = user_analysis.sort_values('Books_Read', ascending=False).reset_index(drop=True)
    user_analysis.index += 1

    ta_col1, ta_col2, ta_col3, ta_col4 = st.columns([2, 2, 1, 1])
    with ta_col1:
        search_uid = st.text_input("Search User ID", placeholder="e.g. 276725", key="ta_search")
    with ta_col2:
        ta_countries = ["ALL"] + sorted(user_analysis['Country'].dropna().unique().tolist())
        ta_country = st.selectbox("Country", ta_countries, key="ta_country")
    with ta_col3:
        ta_min = st.number_input("Min Books", min_value=20, max_value=500, value=20, step=5, key="ta_min")
    with ta_col4:
        ta_rows = st.selectbox("Show", [25, 50, 100, "ALL"], key="ta_rows")

    filtered = user_analysis.copy()
    if search_uid.strip():
        try:
            filtered = filtered[filtered['User-ID'] == int(search_uid.strip())]
        except ValueError:
            pass
    if ta_country != "ALL":
        filtered = filtered[filtered['Country'] == ta_country]
    filtered = filtered[filtered['Books_Read'] >= ta_min]
    total_filtered = len(filtered)
    if ta_rows != "ALL":
        filtered = filtered.head(int(ta_rows))

    st.markdown(f"""
    <div class="summary-bar">
        <span style="color:#3d342a;">Total readers: <b>{len(user_analysis):,}</b></span>
        <span style="color:#c2714f;">Filtered: <b>{total_filtered:,}</b></span>
        <span style="color:#4a7a50;">Showing: <b>{len(filtered):,}</b></span>
        <span style="color:#9a7e2a;">Top reader: <b>#{int(user_analysis.iloc[0]['User-ID'])} ({int(user_analysis.iloc[0]['Books_Read'])} books)</b></span>
    </div>
    """, unsafe_allow_html=True)

    def rating_bar(val, max_val=10):
        pct = int((val / max_val) * 100)
        return (f'<div style="display:flex;align-items:center;gap:.5rem;">'
                f'<div style="width:60px;height:3px;background:#ede8de;border-radius:2px;">'
                f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#6b8f71,#c9a84c);border-radius:2px;"></div>'
                f'</div><span style="color:#3d342a;font-size:.8rem;font-weight:500;">{val}</span></div>')

    def books_bar(val, max_val):
        pct = int((val / max_val) * 100)
        return (f'<div style="display:flex;align-items:center;gap:.5rem;">'
                f'<div style="width:70px;height:3px;background:#ede8de;border-radius:2px;">'
                f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#c2714f,#c9a84c);border-radius:2px;"></div>'
                f'</div><span style="color:#3d342a;font-size:.8rem;font-weight:500;">{val}</span></div>')

    max_books  = user_analysis['Books_Read'].max()
    table_rows = ""
    for rank, (_, row) in enumerate(filtered.iterrows(), 1):
        uid       = int(row['User-ID'])
        books     = int(row['Books_Read'])
        ratings   = int(row['Total_Ratings'])
        avg       = float(row['Avg_Rating'])
        country   = str(row['Country'])[:22]
        rc        = "#b8900a" if rank==1 else "#808080" if rank==2 else "#b0471e" if rank==3 else "#8a7e72"
        row_bg    = "rgba(26,20,16,.03)" if rank%2==0 else "transparent"

        table_rows += f"""
        <tr style="background:{row_bg};border-bottom:1px solid rgba(26,20,16,.07);transition:background .15s;"
            onmouseover="this.style.background='rgba(194,113,79,.07)'"
            onmouseout="this.style.background='{row_bg}'">
            <td style="padding:.55rem .8rem;font-family:'Playfair Display',Georgia,serif;
                       font-size:.88rem;font-weight:700;color:{rc};text-align:center;">#{rank}</td>
            <td style="padding:.55rem .8rem;font-family:'DM Mono','Courier New',monospace;cursor:pointer;
                       font-size:.8rem;color:#3a6090;font-weight:600;"
                onclick="navigator.clipboard.writeText('{uid}')" title="Click to copy">{uid}</td>
            <td style="padding:.55rem .8rem;font-family:'Inter','Segoe UI',sans-serif;font-size:.78rem;color:#5a4e42;">
                📍 {country}</td>
            <td style="padding:.55rem .8rem;font-family:'DM Mono',monospace;font-size:.72rem;">
                {books_bar(books, max_books)}</td>
            <td style="padding:.55rem .8rem;font-family:'DM Mono','Courier New',monospace;font-size:.75rem;
                       color:#5a4e42;text-align:center;font-weight:500;">{ratings:,}</td>
            <td style="padding:.55rem .8rem;font-family:'DM Mono',monospace;font-size:.72rem;">
                {rating_bar(avg)}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;overflow-y:auto;max-height:420px;
                border:1px solid rgba(26,20,16,.12);border-radius:4px;margin-top:.6rem;">
        <table style="width:100%;border-collapse:collapse;font-size:.8rem;background:#faf7f2;">
            <thead>
                <tr style="background:#ede8de;border-bottom:2px solid rgba(194,113,79,.35);
                           position:sticky;top:0;z-index:2;">
                    <th style="padding:.65rem .8rem;font-family:'DM Mono','Courier New',monospace;font-size:.58rem;
                               letter-spacing:.12em;color:#5a4e42;text-align:center;text-transform:uppercase;font-weight:700;">Rank</th>
                    <th style="padding:.65rem .8rem;font-family:'DM Mono','Courier New',monospace;font-size:.58rem;
                               letter-spacing:.12em;color:#3a6090;text-transform:uppercase;font-weight:700;">User ID
                        <span style="color:#8a7e72;font-size:.52rem;"> (click to copy)</span></th>
                    <th style="padding:.65rem .8rem;font-family:'DM Mono','Courier New',monospace;font-size:.58rem;
                               letter-spacing:.12em;color:#5a4e42;text-transform:uppercase;font-weight:700;">Country</th>
                    <th style="padding:.65rem .8rem;font-family:'DM Mono','Courier New',monospace;font-size:.58rem;
                               letter-spacing:.12em;color:#b0471e;text-transform:uppercase;font-weight:700;">Books Read</th>
                    <th style="padding:.65rem .8rem;font-family:'DM Mono','Courier New',monospace;font-size:.58rem;
                               letter-spacing:.12em;color:#5a4e42;text-transform:uppercase;text-align:center;font-weight:700;">Ratings</th>
                    <th style="padding:.65rem .8rem;font-family:'DM Mono','Courier New',monospace;font-size:.58rem;
                               letter-spacing:.12em;color:#4a7a50;text-transform:uppercase;font-weight:700;">Avg Rating</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>
    <div style="font-family:'DM Mono','Courier New',monospace;font-size:.6rem;color:#8a7e72;
                margin-top:.5rem;letter-spacing:.08em;font-weight:500;">
        Click any User ID to copy · Paste into sidebar ·
        Post-EDA filtered (≥ 20 ratings/user · ≥ 5 ratings/book)
    </div>
    """, unsafe_allow_html=True)


# ── BookGalaxy Showcase ──
with st.expander("📚  BookGalaxy Showcase  —  Popular books in dataset with covers", expanded=False):

    st.markdown("""
    <div style="font-family:'Playfair Display',Georgia,serif;font-size:1rem;font-weight:700;
                color:#1a1410;margin-bottom:.3rem;">
        Book Intelligence Catalogue
        <span style="font-family:'DM Mono','Courier New',monospace;font-weight:500;font-size:.65rem;
                     color:#7a6e62;letter-spacing:.05em;margin-left:.8rem;">
            Sorted by unique readers · Post-EDA filtered books only
        </span>
    </div>
    """, unsafe_allow_html=True)

    book_analysis = df.groupby('Book-Title').agg(
        Total_Readers=('User-ID',     'nunique'),
        Total_Ratings=('Book-Rating', 'count'),
        Avg_Rating=   ('Book-Rating', 'mean'),
    ).reset_index()
    book_analysis['Avg_Rating'] = book_analysis['Avg_Rating'].round(2)
    book_info = df[['Book-Title','Book-Author','Publisher','Image-URL-M']].drop_duplicates('Book-Title')
    book_analysis = book_analysis.merge(book_info, on='Book-Title', how='left')
    book_analysis = book_analysis.sort_values('Total_Readers', ascending=False).reset_index(drop=True)
    book_analysis.index += 1

    ba_col1, ba_col2, ba_col3, ba_col4 = st.columns([3, 2, 1, 1])
    with ba_col1:
        ba_search = st.text_input("Search Title / Author", placeholder="e.g. Harry Potter", key="ba_search")
    with ba_col2:
        ba_min_readers = st.slider("Min Readers", 1, 200, 5, key="ba_min_readers")
    with ba_col3:
        ba_min_rating = st.slider("Min Avg Rating", 1.0, 10.0, 1.0, 0.5, key="ba_min_rating")
    with ba_col4:
        ba_rows = st.selectbox("Show", [20, 50, 100, "ALL"], key="ba_rows")

    ba_filtered = book_analysis.copy()
    if ba_search.strip():
        q = ba_search.strip().lower()
        ba_filtered = ba_filtered[
            ba_filtered['Book-Title'].str.lower().str.contains(q, na=False) |
            ba_filtered['Book-Author'].str.lower().str.contains(q, na=False)
        ]
    ba_filtered = ba_filtered[ba_filtered['Total_Readers'] >= ba_min_readers]
    ba_filtered = ba_filtered[ba_filtered['Avg_Rating']    >= ba_min_rating]
    total_ba = len(ba_filtered)
    if ba_rows != "ALL":
        ba_filtered = ba_filtered.head(int(ba_rows))

    st.markdown(f"""
    <div class="summary-bar">
        <span style="color:#3d342a;">Total books: <b>{len(book_analysis):,}</b></span>
        <span style="color:#b0471e;">Filtered: <b>{total_ba:,}</b></span>
        <span style="color:#4a7a50;">Showing: <b>{len(ba_filtered):,}</b></span>
        <span style="color:#9a7e2a;">Most read: <b>{book_analysis.iloc[0]['Book-Title'][:38]}… ({int(book_analysis.iloc[0]['Total_Readers'])} readers)</b></span>
    </div>
    """, unsafe_allow_html=True)

    def reader_bar(val, max_val):
        pct = max(1, int((val / max_val) * 100))
        return (f'<div style="display:flex;align-items:center;gap:.5rem;">'
                f'<div style="width:70px;height:3px;background:#ede8de;border-radius:2px;">'
                f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#c2714f,#c9a84c);border-radius:2px;"></div>'
                f'</div><span style="color:#3d342a;font-size:.8rem;font-weight:500;">{val}</span></div>')

    def avg_bar(val, max_val=10):
        pct = max(1, int((val / max_val) * 100))
        return (f'<div style="display:flex;align-items:center;gap:.5rem;">'
                f'<div style="width:60px;height:3px;background:#ede8de;border-radius:2px;">'
                f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#6b8f71,#5b7fa6);border-radius:2px;"></div>'
                f'</div><span style="color:#3d342a;font-size:.8rem;font-weight:500;">{val}</span></div>')

    max_readers  = book_analysis['Total_Readers'].max()
    ba_rows_html = ""
    for rank, (_, row) in enumerate(ba_filtered.iterrows(), 1):
        title     = str(row['Book-Title'])
        author    = str(row['Book-Author'])
        publisher = str(row.get('Publisher',''))[:28]
        readers   = int(row['Total_Readers'])
        ratings   = int(row['Total_Ratings'])
        avg       = float(row['Avg_Rating'])
        img_url   = str(row.get('Image-URL-M',''))
        rc        = "#b8900a" if rank==1 else "#808080" if rank==2 else "#b0471e" if rank==3 else "#8a7e72"
        row_bg    = "rgba(26,20,16,.03)" if rank%2==0 else "transparent"

        img_cell = (f'<img src="{img_url}" width="48" height="68" '
                    f'style="border-radius:3px;object-fit:cover;display:block;'
                    f'box-shadow:2px 3px 10px rgba(26,20,16,.2);" '
                    f'onerror="this.style.display=\'none\'">'
                    if img_url and img_url != 'nan' else
                    '<div style="width:48px;height:68px;background:#ede8de;border-radius:3px;'
                    'display:flex;align-items:center;justify-content:center;'
                    'font-size:.62rem;color:#8a7e72;font-family:\'Courier New\',monospace;">N/A</div>')

        title_short  = title[:48]  + ('…' if len(title)>48  else '')
        author_short = author[:28] + ('…' if len(author)>28 else '')

        ba_rows_html += f"""
        <tr style="background:{row_bg};border-bottom:1px solid rgba(26,20,16,.07);transition:background .15s;"
            onmouseover="this.style.background='rgba(194,113,79,.06)'"
            onmouseout="this.style.background='{row_bg}'">
            <td style="padding:.5rem .7rem;font-family:'Playfair Display',Georgia,serif;
                       font-size:.88rem;font-weight:700;color:{rc};text-align:center;">#{rank}</td>
            <td style="padding:.5rem .7rem;text-align:center;">{img_cell}</td>
            <td style="padding:.5rem .7rem;">
                <div style="font-family:'Playfair Display',Georgia,serif;font-size:.92rem;
                            font-weight:700;color:#1a1410;line-height:1.3;"
                     title="{title}">{title_short}</div>
                <div style="font-family:'Inter','Segoe UI',sans-serif;font-size:.74rem;
                            color:#b0471e;margin-top:.2rem;font-weight:600;">by {author_short}</div>
                <div style="font-family:'DM Mono','Courier New',monospace;font-size:.6rem;
                            color:#8a7e72;margin-top:.15rem;">{publisher}</div>
            </td>
            <td style="padding:.5rem .7rem;font-family:'DM Mono',monospace;font-size:.72rem;">
                {reader_bar(readers, max_readers)}</td>
            <td style="padding:.5rem .7rem;font-family:'DM Mono','Courier New',monospace;font-size:.75rem;
                       color:#5a4e42;text-align:center;font-weight:500;">{ratings:,}</td>
            <td style="padding:.5rem .7rem;font-family:'DM Mono',monospace;font-size:.72rem;">
                {avg_bar(avg)}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;overflow-y:auto;max-height:480px;
                border:1px solid rgba(26,20,16,.12);border-radius:4px;margin-top:.6rem;">
        <table style="width:100%;border-collapse:collapse;background:#faf7f2;">
            <thead>
                <tr style="background:#ede8de;border-bottom:2px solid rgba(194,113,79,.35);
                           position:sticky;top:0;z-index:2;">
                    <th style="padding:.65rem .7rem;font-family:'DM Mono','Courier New',monospace;font-size:.56rem;
                               letter-spacing:.12em;color:#5a4e42;text-align:center;
                               min-width:50px;text-transform:uppercase;font-weight:700;">Rank</th>
                    <th style="padding:.65rem .7rem;font-family:'DM Mono','Courier New',monospace;font-size:.56rem;
                               letter-spacing:.12em;color:#9a7e2a;text-align:center;
                               min-width:60px;text-transform:uppercase;font-weight:700;">Cover</th>
                    <th style="padding:.65rem .7rem;font-family:'DM Mono','Courier New',monospace;font-size:.56rem;
                               letter-spacing:.12em;color:#1a1410;min-width:220px;text-transform:uppercase;font-weight:700;">Title / Author</th>
                    <th style="padding:.65rem .7rem;font-family:'DM Mono','Courier New',monospace;font-size:.56rem;
                               letter-spacing:.12em;color:#b0471e;min-width:130px;text-transform:uppercase;font-weight:700;">Readers</th>
                    <th style="padding:.65rem .7rem;font-family:'DM Mono','Courier New',monospace;font-size:.56rem;
                               letter-spacing:.12em;color:#5a4e42;text-align:center;
                               min-width:90px;text-transform:uppercase;font-weight:700;">Ratings</th>
                    <th style="padding:.65rem .7rem;font-family:'DM Mono','Courier New',monospace;font-size:.56rem;
                               letter-spacing:.12em;color:#4a7a50;min-width:120px;text-transform:uppercase;font-weight:700;">Avg Rating</th>
                </tr>
            </thead>
            <tbody>{ba_rows_html}</tbody>
        </table>
    </div>
    <div style="font-family:'DM Mono','Courier New',monospace;font-size:.6rem;color:#8a7e72;
                margin-top:.5rem;letter-spacing:.08em;font-weight:500;">
        Sorted by unique readers · Covers from book database · Post-EDA filtered (≥ 5 ratings/book)
    </div>
    """, unsafe_allow_html=True)


# ── Execute ──
if run_btn:
    user_id = int(user_id_input)

    if user_id not in df['User-ID'].values:
        st.markdown(f"""
        <div style="padding:1.2rem 1.4rem;background:#fdf0ec;
                    border-left:4px solid #c2714f;border-radius:0 4px 4px 0;
                    font-family:'Inter','Segoe UI',sans-serif;font-size:.88rem;color:#8a2800;">
            User ID <b>{user_id}</b> was not found in the dataset.
            Please use <em>The User Storyboard</em> above to find a valid ID.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    user_info  = df[df['User-ID'] == user_id].iloc[0]
    books_read = df[df['User-ID'] == user_id]['Book-Title'].nunique()
    avg_rating = df[df['User-ID'] == user_id]['Book-Rating'].mean()
    country    = user_info.get('Country', 'Unknown')

    st.markdown(f"""
    <div class="user-chip">
        <div class="user-dot"></div>
        Reader #{user_id} &nbsp;·&nbsp; {country} &nbsp;·&nbsp;
        {books_read} books read &nbsp;·&nbsp; avg rating {avg_rating:.1f}
    </div>
    """, unsafe_allow_html=True)

    # Processing log
    log_placeholder = st.empty()
    logs = [
        f"Initialising recommendation engine for reader {user_id}…",
        "Computing item-based similarity vectors…",
        "Fetching top-10 similar readers…",
        f"Merging signals — item {item_weight:.0%} · user {user_weight:.0%}…",
        f"Generating top {n_recs} personalised recommendations…",
        "Done. Rendering your reading list.",
    ]
    for i, log in enumerate(logs):
        log_html = "".join(
            f'<div class="log-line"><span style="color:#5a9060;margin-right:.8rem;">[{j+1:02d}]</span>{l}</div>'
            for j, l in enumerate(logs[:i+1])
        )
        log_placeholder.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)
        time.sleep(0.25)
    time.sleep(0.3)
    log_placeholder.empty()

    with st.spinner("Finding your next great reads…"):
        results = recommend_for_user(
            user_id, df, item_sim_df, user_sim_df,
            item_weight, user_weight, n=n_recs
        )

    if not results:
        st.markdown("""
        <div class="empty-state">
            No recommendations found.<br>
            <span style="font-size:.88rem;opacity:.8;">
                Try a different User ID or adjust the weight parameters.
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown(f'<div class="section-badge">✦ Your Top {len(results)} Reads · Reader {user_id}</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:1.2rem;">
        <span class="tag tag-item">Item ×{item_weight}</span>
        <span class="tag tag-user">User ×{user_weight}</span>
        <span class="tag tag-hybrid">Hybrid</span>
    </div>
    """, unsafe_allow_html=True)

    max_score = max(r['Score'] for r in results) or 1

    for i, book in enumerate(results, 1):
        bar_pct    = int((book['Score'] / max_score) * 100)
        rank_color = "#b8900a" if i==1 else "#808080" if i==2 else "#b0471e" if i==3 else "#8a7e72"
        img_tag    = (f'<img src="{book["Image"]}" width="60" height="85" class="book-cover" '
                      f'onerror="this.style.display=\'none\'">' if book.get("Image") else "")

        title_display = book['Book-Title'][:72] + ('…' if len(book['Book-Title']) > 72 else '')

        st.markdown(f"""
        <div class="book-card">
            <div class="book-rank" style="color:{rank_color};">#{i:02d}</div>
            {img_tag}
            <div class="book-info">
                <div class="book-title">{title_display}</div>
                <div class="book-author">by {book['Author']}</div>
                <div class="book-publisher">{book['Publisher']}</div>
            </div>
            <div class="book-score">
                <div style="font-size:.92rem;font-weight:700;color:#1a1410;">{book['Score']:.3f}</div>
                <div style="font-size:.58rem;color:#7a6e62;margin-top:.2rem;letter-spacing:.08em;font-weight:600;">
                    MATCH SCORE
                </div>
                <div class="score-bar" style="width:{bar_pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:2.5rem;padding:1.2rem 0;
                border-top:1px solid rgba(26,20,16,.12);
                font-family:'Playfair Display',Georgia,serif;font-size:.88rem;
                color:#7a6e62;font-style:italic;">
        PageTurn · Hybrid Collaborative Filtering ·
        {len(results)} recommendations for reader {user_id}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="position:relative;margin:1.5rem 0;overflow:hidden;
                border-radius:6px;border:1px solid rgba(26,20,16,.12);">
        <img src="https://images.unsplash.com/photo-1512820790803-83ca734da794?w=1400&q=70"
             style="width:100%;height:280px;object-fit:cover;opacity:.55;display:block;">
        <div style="position:absolute;inset:0;
                    background:linear-gradient(135deg,rgba(245,240,232,.82) 0%,rgba(240,235,224,.55) 100%);
                    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.9rem;">
            <div style="font-family:'Playfair Display',Georgia,serif;
                        font-size:clamp(1.2rem,3vw,1.9rem);font-weight:900;
                        color:#1a1410;letter-spacing:-.01em;text-align:center;
                        text-shadow:0 2px 6px rgba(255,255,255,.6);">
                Your next great read is one click away.
            </div>
            <div style="font-family:'Inter','Segoe UI',sans-serif;font-size:.88rem;
                        color:#4a3e32;letter-spacing:.02em;font-weight:500;">
                Open <em>The User Storyboard</em> · find your ID · click <em>Find My Books</em>
            </div>
            <div style="display:flex;gap:.6rem;margin-top:.3rem;flex-wrap:wrap;justify-content:center;">
                <span style="font-family:'DM Mono','Courier New',monospace;font-size:.65rem;padding:.35rem .9rem;
                             border-radius:3px;background:rgba(194,113,79,.12);
                             border:1px solid rgba(194,113,79,.35);color:#a0451e;letter-spacing:.08em;font-weight:600;">
                    Hybrid CF</span>
                <span style="font-family:'DM Mono','Courier New',monospace;font-size:.65rem;padding:.35rem .9rem;
                             border-radius:3px;background:rgba(91,127,166,.12);
                             border:1px solid rgba(91,127,166,.35);color:#2a5a90;letter-spacing:.08em;font-weight:600;">
                    Cosine Similarity</span>
                <span style="font-family:'DM Mono','Courier New',monospace;font-size:.65rem;padding:.35rem .9rem;
                             border-radius:3px;background:rgba(201,168,76,.12);
                             border:1px solid rgba(201,168,76,.35);color:#8a6a10;letter-spacing:.08em;font-weight:600;">
                    Smart Ranking</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)