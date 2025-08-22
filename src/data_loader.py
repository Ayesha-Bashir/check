import re
import pandas as pd

YEAR_RE = r"\((\d{4})\)"


def _extract_year(title):
    m = re.search(YEAR_RE, str(title))
    return int(m.group(1)) if m else None


def load_movie_catalog(csv_path: str = "data/movies.csv", limit: int | None = None):
    """
    Loads a MovieLens CSV with headers: movieId,title,genres.
    Returns:
      records: list of dicts with keys (movie_id, title, genre, poster_url)
      id_to_movie: dict mapping int(movie_id) → record
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Rename columns to canonical names
    colmap = {}
    for cand in ["movie_id", "movieId", "id"]:
        if cand in df.columns:
            colmap[cand] = "movie_id"
            break
    for cand in ["title", "Title", "movie_title", "name"]:
        if cand in df.columns:
            colmap[cand] = "title"
            break
    for cand in ["genre", "genres", "Genre", "Genres"]:
        if cand in df.columns:
            colmap[cand] = "genre"
            break
    for cand in ["poster_url", "poster", "image", "posterUrl", "Poster"]:
        if cand in df.columns:
            colmap[cand] = "poster_url"
            break

    if "movie_id" not in colmap.values():
        raise ValueError("No movie id column found (movieId/id).")
    if "title" not in colmap.values():
        raise ValueError("No title column found (title).")

    df = df.rename(columns=colmap)
    keep = [c for c in ["movie_id", "title",
                        "genre", "poster_url"] if c in df.columns]
    df = df[keep].copy()

    # Coerce ids
    df["movie_id"] = pd.to_numeric(
        df["movie_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["movie_id"]).copy()
    df["movie_id"] = df["movie_id"].astype(int)

    # Fill missing optional cols
    if "genre" not in df.columns:
        df["genre"] = ""
    if "poster_url" not in df.columns:
        df["poster_url"] = ""

    # Add year column
    df["year"] = df["title"].apply(_extract_year)

    if limit is not None:
        df = df.iloc[:limit]

    records = df.to_dict(orient="records")
    id_to_movie = {int(r["movie_id"]): r for r in records}
    return records, id_to_movie


def map_recommendations(recs, id_lookup):
    """
    Map a list of (movie_id, rating) to display dicts using the catalog lookup.
    Handles 0-based vs 1-based ids defensively.
    """
    mapped = []
    for mid, rating in recs:
        mid_int = int(mid)
        if mid_int not in id_lookup and (mid_int + 1) in id_lookup:
            mid_int += 1
        movie = id_lookup.get(
            mid_int, {"title": f"Movie {mid_int}", "genre": "", "poster_url": ""})
        mapped.append({
            "title": movie["title"],
            "genre": movie.get("genre", ""),
            "poster": movie.get("poster_url", ""),
            "pred": round(float(rating), 2) if rating is not None else None,
        })
    return mapped


def load_links(csv_path="data/links.csv"):
    """
    Returns {movieId:int -> tmdbId:int or None}
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    # expected columns: movieId, imdbId, tmdbId
    if "movieId" not in df.columns:
        raise ValueError("links.csv missing movieId")
    tmdb_col = "tmdbId" if "tmdbId" in df.columns else None
    mapping = {}
    for _, row in df.iterrows():
        mid = int(row["movieId"])
        tmid = int(row[tmdb_col]) if tmdb_col and not pd.isna(
            row[tmdb_col]) else None
        mapping[mid] = tmid
    return mapping
