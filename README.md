# Sistem Rekomendasi Film — MovieLens Latest Small

Aplikasi rekomendasi film berbasis konten menggunakan Python, Pandas, Scikit-learn, TF-IDF, Cosine Similarity, dan Streamlit. Tidak membutuhkan TMDb API Key maupun akses API eksternal.

## Fitur
- Memuat katalog 9.742 film dari MovieLens Latest Small.
- Menggabungkan genre dan tag pengguna sebagai fitur konten.
- Menghitung kemiripan dengan TF-IDF dan Cosine Similarity secara sparse (tidak membuat matriks penuh NxN).
- Menghasilkan hingga 10 rekomendasi, mengecualikan film sumber, dan menjelaskan genre/tag yang sama.
- Opsi penyeimbang rating menggunakan Bayesian shrinkage.
- Tautan ke halaman TMDb dari `tmdbId` di `links.csv` bila tersedia.

## Struktur dataset
Letakkan empat file CSV di `data/raw/ml-latest-small/`:
`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`.
File-file tersebut sudah disertakan pada ZIP proyek.

## Instalasi (Windows / VS Code)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Buka alamat lokal yang ditampilkan Streamlit (biasanya `http://localhost:8501`).

Jika PowerShell menolak aktivasi environment, jalankan:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Menjalankan pengujian
```powershell
python -m pytest -q
```

## Catatan poster
Dataset MovieLens tidak menyertakan file poster atau URL poster. `links.csv` menyediakan `tmdbId`, sehingga aplikasi menyediakan tautan ke halaman film TMDb. Poster otomatis di dalam aplikasi membutuhkan dataset poster terpisah atau layanan eksternal; fitur rekomendasi inti tetap berjalan offline.

## Dataset dan atribusi
MovieLens Latest Small oleh GroupLens Research, University of Minnesota. Dataset berisi metadata film, rating, tag, dan tautan eksternal. Harap pertahankan atribusi dan ketentuan penggunaan yang tercantum pada `data/raw/ml-latest-small/README.txt`.

Referensi: Harper, F. Maxwell, and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4): 19:1–19:19. https://doi.org/10.1145/2827872
