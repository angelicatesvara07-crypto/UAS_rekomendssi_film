import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_model(movies_df):
    # Bersihkan format genre (misal: "Action|Comedy" jadi "Action Comedy")
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    
    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    # Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, movies_df, cosine_sim, top_n=10):
    # Cari index film berdasarkan judul
    idx = movies_df[movies_df['title'].str.contains(title, case=False)].index[0]
    
    # Dapatkan skor similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ambil top_n (index 0 adalah film itu sendiri, jadi skip)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices][['title', 'genres']]