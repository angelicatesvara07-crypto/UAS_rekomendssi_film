import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.features import combine_features
from src.recommender import get_recommendations

def precision_at_k(recommended, relevant, k=10):
    rec = recommended[:k]
    hits = sum(1 for r in rec if r in relevant)
    return hits / k if k else 0.0

def compare_variants(movies_df, title, top_n=10):
    variants = {
        "genre_only": (True, False, False, False),
        "genre+overview": (True, True, False, False),
        "all_features": (True, True, True, True),
    }
    results = {}
    for name, (g, o, d, c) in variants.items():
        df = combine_features(movies_df, g, o, d, c)
        tfidf = TfidfVectorizer(stop_words='english')
        mat = tfidf.fit_transform(df['combined_features'])
        sim = cosine_similarity(mat, mat)
        recs = get_recommendations(title, df, sim, top_n=top_n) if False else None
        results[name] = df
    return results
