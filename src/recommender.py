import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.features import combine_features

class ContentRecommender:
    def __init__(self, movies, tags=None, ratings=None):
        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings.copy() if ratings is not None else pd.DataFrame(columns=["movieId", "rating"])
        self.tags = tags.copy() if tags is not None else pd.DataFrame(columns=["movieId", "tag"])
        tag_text = self._aggregate_tags()
        self.df = combine_features(self.movies, tag_text)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1,
                                          max_features=100000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.df["combined_features"])
        self._rating_stats()
        self._title_lookup = {}
        for i, title in enumerate(self.df["title"].astype(str)):
            self._title_lookup.setdefault(title.casefold(), i)

    def _aggregate_tags(self):
        if self.tags.empty or not {"movieId", "tag"}.issubset(self.tags.columns):
            return pd.Series("", index=self.movies.index)
        grouped = self.tags.groupby("movieId")["tag"].apply(lambda s: " ".join(s.astype(str).tolist()))
        return self.movies["movieId"].map(grouped).fillna("")

    def _rating_stats(self):
        if self.ratings.empty or not {"movieId", "rating"}.issubset(self.ratings.columns):
            self.df["avg_rating"] = np.nan
            self.df["rating_count"] = 0
            return
        stats = self.ratings.groupby("movieId")["rating"].agg(avg_rating="mean", rating_count="count")
        self.df = self.df.merge(stats, left_on="movieId", right_index=True, how="left")
        self.df["rating_count"] = self.df["rating_count"].fillna(0).astype(int)

    def find_index(self, title):
        key = str(title).strip().casefold()
        if key in self._title_lookup:
            return self._title_lookup[key]
        matches = self.df.index[self.df.title.astype(str).str.casefold().str.contains(re.escape(key), na=False)]
        if len(matches):
            return int(matches[0])
        raise ValueError(f"Film '{title}' tidak ditemukan. Coba pilih judul dari daftar.")

    def recommend(self, title, top_n=10, use_popularity=True):
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n harus berupa bilangan bulat positif")
        idx = self.find_index(title)
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        scores[idx] = -1.0
        candidate = np.argsort(scores)[::-1]
        candidate = [int(i) for i in candidate if scores[i] >= 0][:max(top_n * 5, top_n)]
        if use_popularity and candidate and self.df["rating_count"].max() > 0:
            counts = self.df.loc[candidate, "rating_count"].to_numpy(dtype=float)
            means = self.df.loc[candidate, "avg_rating"].fillna(0).to_numpy(dtype=float)
            # Bayesian shrinkage reduces the impact of films with only a few ratings.
            global_mean = float(self.ratings["rating"].mean()) if not self.ratings.empty else 3.5
            m = 20.0
            bayes = (counts * means + m * global_mean) / (counts + m)
            quality = bayes / 5.0
            adjusted = np.array([scores[i] for i in candidate]) * (0.85 + 0.15 * quality)
            candidate = [candidate[i] for i in np.argsort(adjusted)[::-1]]
        chosen = candidate[:top_n]
        rows = self.df.iloc[chosen].copy()
        rows["similarity"] = [float(scores[i]) for i in chosen]
        source_genres = set(str(self.df.iloc[idx].get("genres", "")).split("|")) - {"", "(no genres listed)"}
        explanations = []
        for _, row in rows.iterrows():
            target_genres = set(str(row.get("genres", "")).split("|")) - {"", "(no genres listed)"}
            common = sorted(source_genres & target_genres)
            if common:
                explanations.append("Genre yang sama: " + ", ".join(common[:4]))
            else:
                source_tags = set(self.df.iloc[idx]["tags_text"].split())
                target_tags = set(row["tags_text"].split())
                shared = sorted(source_tags & target_tags)
                explanations.append("Tag yang sama: " + ", ".join(shared[:5]) if shared else "Memiliki kemiripan pola genre dan tag.")
        rows["explanation"] = explanations
        rows["rank"] = range(1, len(rows) + 1)
        return rows.reset_index(drop=True)

def build_model(movies_df, use_genres=True, use_overview=False, use_director=False, use_cast=False):
    # Compatibility wrapper for the original project API.
    df = movies_df.copy()
    if not use_genres and "genres" in df:
        df["genres"] = ""
    if use_overview and "overview" in df:
        df["genres"] = df.get("genres", "").astype(str) + " " + df["overview"].fillna("").astype(str)
    if use_director and "director" in df:
        df["genres"] = df.get("genres", "").astype(str) + " " + df["director"].fillna("").astype(str)
    if use_cast and "cast" in df:
        df["genres"] = df.get("genres", "").astype(str) + " " + df["cast"].fillna("").astype(str)
    model = ContentRecommender(df)
    return model.matrix, model.df

def get_recommendations(title, movies_df, cosine_sim, top_n=10):
    matches = movies_df[movies_df["title"].astype(str).str.casefold().eq(str(title).strip().casefold())]
    if matches.empty:
        matches = movies_df[movies_df["title"].astype(str).str.contains(re.escape(str(title)), case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Film '{title}' tidak ditemukan.")
    idx = int(matches.index[0])
    scores = cosine_sim[idx].toarray().ravel() if hasattr(cosine_sim[idx], "toarray") else np.asarray(cosine_sim[idx]).ravel()
    scores[idx] = -1
    ids = np.argsort(scores)[::-1][:top_n]
    out = movies_df.iloc[ids].copy()
    out["similarity"] = [round(float(scores[i]), 4) for i in ids]
    return out.reset_index(drop=True)
