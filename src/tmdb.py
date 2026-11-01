import requests

# Nanti ganti dengan API Key milikmu dari TMDb
API_KEY = "MASUKKAN_API_KEY_KAMU_DISINI"
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_poster(movie_title):
    """
    Fungsi untuk mengambil URL poster film dari TMDb API berdasarkan judul.
    """
    if API_KEY == "MASUKKAN_API_KEY_KAMU_DISINI":
        return "API Key belum diatur"
        
    endpoint = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "query": movie_title
    }
    
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                # URL dasar untuk gambar poster TMDb
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None