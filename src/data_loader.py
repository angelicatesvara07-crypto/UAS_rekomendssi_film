from pathlib import Path
import pandas as pd

REQUIRED = {"movies.csv", "ratings.csv", "tags.csv", "links.csv"}

def load_dataset(data_dir="data/raw/ml-latest-small"):
    root = Path(data_dir)
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        raise FileNotFoundError("File dataset tidak ditemukan: " + ", ".join(missing))
    movies = pd.read_csv(root / "movies.csv")
    ratings = pd.read_csv(root / "ratings.csv")
    tags = pd.read_csv(root / "tags.csv")
    links = pd.read_csv(root / "links.csv")
    for frame, column in [(movies, "movieId"), (ratings, "movieId"), (tags, "movieId"), (links, "movieId")]:
        if column not in frame.columns:
            raise ValueError(f"Kolom wajib '{column}' tidak ada dalam dataset")
    movies = movies.dropna(subset=["title"]).drop_duplicates("movieId").reset_index(drop=True)
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["genres"] = movies.get("genres", pd.Series(index=movies.index, dtype=str)).fillna("").astype(str)
    ratings = ratings.dropna(subset=["movieId", "rating"])
    tags = tags.dropna(subset=["movieId", "tag"])
    tags["tag"] = tags["tag"].astype(str).str.strip()
    return movies, tags, ratings, links
