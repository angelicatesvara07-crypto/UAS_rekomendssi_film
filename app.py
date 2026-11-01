import streamlit as st
import pandas as pd
from src.recommender import build_model, get_recommendations

st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬")

st.title("🎬 Sistem Rekomendasi Film")
st.write("Masukkan film favoritmu, dan sistem akan merekomendasikan 10 film yang mirip berdasarkan konten (Genre).")

# Fungsi untuk load data
@st.cache_data
def load_data():
    try:
        # Sesuaikan path ini nanti kalau datasetnya sudah didownload
        movies = pd.read_csv('data/raw/ml-latest-small/movies.csv')
        return movies
    except FileNotFoundError:
        return None

movies = load_data()

if movies is not None:
    cosine_sim = build_model(movies)
    selected_movie = st.selectbox("Pilih judul film:", movies['title'].values)
    
    if st.button("Cari Rekomendasi"):
        with st.spinner('Mencari film...'):
            recommendations = get_recommendations(selected_movie, movies, cosine_sim)
            st.success(f"Rekomendasi untuk **{selected_movie}**:")
            st.dataframe(recommendations, use_container_width=True)
else:
    st.warning("⚠️ Dataset belum ditemukan. Silakan download MovieLens Latest Small dan taruh di folder `data/raw/ml-latest-small/`.")