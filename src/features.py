import re
import pandas as pd

def _normalise(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("|", " ").lower()).strip()

def combine_features(movies_df, tag_text=None):
    df = movies_df.copy()
    df["genres_text"] = df["genres"].map(_normalise) if "genres" in df else ""
    if tag_text is None:
        tag_text = pd.Series("", index=df.index)
    df["tags_text"] = tag_text.reindex(df.index).fillna("").map(_normalise)
    # Repeat genre tokens to give the relatively reliable catalog metadata more weight.
    df["combined_features"] = (df["genres_text"] + " " + df["genres_text"] + " " + df["tags_text"]).str.strip()
    df.loc[df["combined_features"].eq(""), "combined_features"] = "unknown"
    return df
