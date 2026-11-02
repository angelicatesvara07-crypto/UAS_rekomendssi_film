import streamlit as st
import pandas as pd
from src.data_loader import load_dataset
from src.recommender import ContentRecommender

st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬", layout="wide")
st.title("🎬 Rekomendasi Film")
st.caption("Rekomendasi berbasis genre dan tag MovieLens — tanpa API key atau koneksi API eksternal.")

@st.cache_data(show_spinner="Memuat dataset MovieLens...")
def get_data():
    return load_dataset("data/raw/ml-latest-small")

@st.cache_resource(show_spinner="Membangun model rekomendasi...")
def get_model(_movies, _tags, _ratings):
    return ContentRecommender(_movies, _tags, _ratings)

try:
    movies, tags, ratings, links = get_data()
    model = get_model(movies, tags, ratings)
except Exception as exc:
    st.error(f"Dataset gagal dimuat: {exc}")
    st.info("Pastikan movies.csv, ratings.csv, tags.csv, dan links.csv berada di data/raw/ml-latest-small/.")
    st.stop()

with st.sidebar:
    st.header("Tentang Dataset")
    st.metric("Jumlah film", f"{len(movies):,}")
    st.metric("Rating", f"{len(ratings):,}")
    st.metric("Tag", f"{len(tags):,}")
    st.divider()
    top_n = st.slider("Jumlah rekomendasi", 5, 10, 10)
    use_popularity = st.checkbox("Gunakan rating sebagai penyeimbang", value=True,
                                 help="Membantu mengutamakan film yang memiliki dukungan rating lebih banyak.")

movie_options = movies.sort_values("title")["title"].tolist()
selected = st.selectbox("Pilih film favorit", movie_options, index=None,
                        placeholder="Ketik judul film, misalnya Toy Story...")

if selected:
    chosen = movies.loc[movies.title.eq(selected)].iloc[0]
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(selected)
        st.write(f"**Genre:** {chosen.get('genres', 'Tidak tersedia').replace('|', ', ')}")
    with c2:
        tmdb_id = links.loc[links.movieId.eq(chosen.movieId), "tmdbId"]
        if not tmdb_id.empty and pd.notna(tmdb_id.iloc[0]) and int(tmdb_id.iloc[0]) > 0:
            st.link_button("Buka di TMDb", f"https://www.themoviedb.org/movie/{int(tmdb_id.iloc[0])}")
    if st.button(f"Cari {top_n} Rekomendasi", type="primary", use_container_width=True):
        with st.spinner("Menghitung kemiripan film..."):
            recs = model.recommend(selected, top_n=top_n, use_popularity=use_popularity)
        st.session_state["recommendations"] = recs
        st.session_state["selected_movie"] = selected

if "recommendations" in st.session_state and st.session_state.get("selected_movie") == selected:
    st.divider()
    st.subheader(f"Film yang mirip dengan {selected}")
    recs = st.session_state["recommendations"]
    if recs.empty:
        st.warning("Belum ditemukan rekomendasi untuk film tersebut.")
    else:
        for start in range(0, len(recs), 2):
            cols = st.columns(2)
            for col, (_, row) in zip(cols, recs.iloc[start:start+2].iterrows()):
                with col:
                    st.markdown(f"#### {int(row['rank'])}. {row['title']}")
                    genres = str(row.get("genres", "")).replace("|", ", ")
                    st.caption(f"Genre: {genres or 'Tidak diketahui'}")
                    st.progress(float(max(0, min(1, row['similarity']))),
                                text=f"Kemiripan konten: {row['similarity']:.1%}")
                    if row.get("explanation"):
                        st.write(f"**Alasan:** {row['explanation']}")
                    if pd.notna(row.get("avg_rating")):
                        st.caption(f"Rating rata-rata: {row['avg_rating']:.2f}/5 · {int(row['rating_count']):,} penilaian")
                    tmdb_id = row.get("tmdbId")
                    if pd.notna(tmdb_id) and int(tmdb_id) > 0:
                        st.markdown(f"[Lihat detail film di TMDb](https://www.themoviedb.org/movie/{int(tmdb_id)})")
            st.divider()

st.caption("Sumber: MovieLens Latest Small (GroupLens). Poster tidak disimpan dalam dataset ini; tautan TMDb tersedia bila ID film ditemukan.")
