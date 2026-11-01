import pandas as pd

def clean_genres(movies_df):
    """
    Membersihkan kolom genre dari format 'Action|Comedy' menjadi 'Action Comedy'
    agar bisa dibaca oleh TF-IDF.
    """
    if 'genres' in movies_df.columns:
        movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    return movies_df