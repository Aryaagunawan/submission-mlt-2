# Laporan Proyek Machine Learning - Arya Gunawan

## Project Overview

Proyek ini berfokus pada analisis dan prediksi popularitas lagu menggunakan data musik dari Spotify. Dengan lebih dari 30.000 entri lagu, proyek ini bertujuan untuk memahami faktor-faktor yang mempengaruhi popularitas lagu berdasarkan fitur audio seperti danceability, energy, tempo, dan lainnya.

Masalah ini penting untuk diselesaikan karena dalam industri musik digital, memahami preferensi pengguna dan karakteristik lagu yang populer dapat membantu musisi, label rekaman, dan platform streaming untuk merancang strategi distribusi dan rekomendasi yang lebih efektif.

Sistem rekomendasi berperan dalam meningkatkan pengalaman pengguna dengan memanfaatkan data seperti preferensi lagu, riwayat pemutaran, dan konten lagu itu sendiri. Salah satu metode yang diterapkan adalah Content-based Filtering, yaitu merekomendasikan lagu berdasarkan kesamaan fitur musik seperti tempo, energy, dan danceability serta aspek teks dari judul lagu. Dalam prosesnya, TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk memberikan bobot pada setiap kata dalam judul lagu, sementara Cosine Similarity dipakai untuk mengukur tingkat kemiripan antar lagu berdasarkan fitur musik dan judul tersebut.

Pendekatan ini memungkinkan sistem untuk menyajikan 10 lagu rekomendasi teratas yang sesuai dengan preferensi pengguna. Hasil evaluasi menunjukkan bahwa sistem memiliki performa yang baik dengan akurasi tinggi, menandakan bahwa rekomendasi yang diberikan relevan berdasarkan analisis kesamaan konten lagu.



**Referensi**:
- [Content-driven Music Recommendation: Evolution, State of the Art, and Challenges](https://arxiv.org/abs/2107.11803)
- [Learning content similarity for music recommendation](https://arxiv.org/abs/1105.2344)

## Business Understanding

Jumlah lagu yang sangat banyak di platform streaming sering menyulitkan pengguna menemukan lagu sesuai selera. Untuk mengatasi hal ini, dibutuhkan sistem rekomendasi yang mampu memberikan saran lagu yang relevan berdasarkan preferensi pengguna, tanpa harus bergantung pada riwayat pemutaran.

Bagian laporan ini mencakup:

### Problem Statements

- Bagaimana membantu pengguna menemukan lagu sesuai preferensi secara efisien?
- Bagaimana memberikan rekomendasi lagu tanpa bergantung pada histori pengguna?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Memberikan pengalaman personal melalui rekomendasi lagu yang relevan.
- Membangun sistem rekomendasi berbasis konten menggunakan fitur audio dan teks lagu.

### Solution statements
- Menggunakan Content-Based Filtering untuk menganalisis fitur lagu.
- Menerapkan TF-IDF pada judul lagu dan menghitung kemiripan dengan Cosine Similarity.
- Collaborative Filtering bisa digunakan jika data histori pemutaran tersedia, atau hybrid approach yang menggabungkan histori dengan konten musik untuk rekomendasi yang lebih akurat.

## Data Understanding

Dataset yang digunakan berasal dari kaggle [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs/data). Dataset ini berisi 32833 baris dan 23 kolom yang mencakup metadata dan fitur numerik audio dari Spotify API. Fitur-Fitur:

###  Kolom dan Deskripsi

| Kolom                      | Deskripsi                                                                 |
|----------------------------|---------------------------------------------------------------------------|
| `track_id`                 | ID unik lagu                                                              |
| `track_name`               | Nama lagu                                                                 |
| `track_artist`             | Nama artis                                                                |
| `track_popularity`         | Popularitas lagu (skala 0–100)                                            |
| `track_album_id`           | ID album lagu                                                             |
| `track_album_name`         | Nama album lagu                                                           |
| `track_album_release_date` | Tanggal rilis lagu                                                        |
| `playlist_id`              | ID playlist asal                                                          |
| `playlist_name`            | Nama playlist                                                             |
| `playlist_genre`           | Genre playlist                                                            |
| `playlist_subgenre`        | Subgenre playlist                                                         |
| `danceability`             | Seberapa mudah lagu digunakan untuk menari (0–1)                          |
| `energy`                   | Energi keseluruhan lagu (0–1)                                             |
| `key`                      | Tangga nada lagu (0–11)                                                   |
| `loudness`                 | Tingkat kerasnya suara (dalam desibel)                                    |
| `mode`                     | Modus lagu (`1` = mayor, `0` = minor)                                     |
| `speechiness`              | Porsi lirik berbicara dalam lagu (0–1)                                    |
| `acousticness`             | Tingkat keakustikan lagu (0–1)                                            |
| `instrumentalness`         | Kemungkinan lagu tidak memiliki vokal (0–1)                               |
| `liveness`                 | Indikasi apakah lagu direkam di konser langsung (0–1)                     |
| `valence`                  | Tingkat keceriaan lagu (0–1)                                              |
| `tempo`                    | Tempo lagu (dalam Beats Per Minute / BPM)                                 |
| `duration_ms`              | Durasi lagu (dalam milidetik)                                             |


### Exploratory Data Analysis (EDA)

**Distribusi Popularitas Lagu**

```python
plt.figure(figsize=(10, 5))
sns.histplot(df_song['track_popularity'], bins=30, kde=True, color='green')
plt.title('Distribusi Popularitas Lagu')
plt.xlabel('Skor Popularitas')
plt.ylabel('Jumlah Lagu')
plt.show()
```

![image](https://github.com/user-attachments/assets/be6e9291-570b-414d-b0a4-d430013139ab)

Insight:

Sebagian besar lagu sangat tidak populer (skor 0), dengan distribusi lain yang tersebar antara skor 20-80, menunjukkan beberapa puncak popularitas. Sangat sedikit lagu yang mencapai popularitas tinggi.


**Korelasi Antar Fitur Numerik**

```python
plt.figure(figsize=(14, 10))
corr = df_song.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Heatmap Korelasi Antar Fitur Numerik')
plt.show()
```

![image](https://github.com/user-attachments/assets/d3541900-31bf-493d-bb42-60df2aeaea7d)

Insight:

Korelasi fitur track_popularity sangat lemah dengan semua fitur lain. Energy dan loudness berkorelasi positif kuat, sedangkan acousticness dan energy berkorelasi negatif kuat.



## Data Preparation
Tahap Data Preparation dilakukan untuk memastikan data yang digunakan bersih, konsisten, dan siap digunakan dalam proses analisis serta pemodelan machine learning. Berikut adalah tahapan yang dilakukan secara berurutan:

1. Menghapus Data Kosong (Missing Values)
   Beberapa baris dalam dataset memiliki nilai kosong (NaN) yang berpotensi mengganggu proses analisis dan pelatihan model. Oleh karena itu, seluruh baris yang memiliki nilai kosong dihapus menggunakan fungsi dropna().
```python
df_song.dropna(inplace=True)
```
  Alasannya Menghapus data kosong mencegah error dalam perhitungan dan menjaga integritas data.
  
2. Menghapus Data Duplikat
   Beberapa entri dalam dataset memiliki ID lagu (track_id) yang sama, yang berarti terjadi duplikasi data. Untuk menghindari redundansi dan bias pada model, data duplikat dihapus berdasarkan kolom track_id.
```python
df_song = df_song.drop_duplicates(subset='track_id').reset_index(drop=True)
```
   Hasil: Jumlah data setelah proses pembersihan adalah 28.352 baris dan 23 kolom.

3. Menampilkan Genre dan Subgenre
   Proses ini bertujuan untuk memahami distribusi genre dan subgenre yang tersedia dalam dataset.
```python
for genre in df_song['playlist_genre'].unique():
    subgenres = df_song['playlist_subgenre'][df_song['playlist_genre'] == genre].unique().tolist()
    print(f"{genre}: {subgenres}")
```
   Insight: Dataset memiliki genre utama seperti pop, rap, rock, latin, r&b, dan edm, masing-masing dengan subgenre yang cukup beragam.

4. Menentukan Album Paling Populer per Artis
   Langkah ini bertujuan mengevaluasi album mana yang paling populer untuk setiap artis, dengan menghitung rata-rata popularitas lagu dalam album.
```python
# Mengelompokkan data berdasarkan artis dan album, lalu menghitung rata-rata popularitas lagu per album
album_popularity = df_song.groupby(
    ['track_artist', 'track_album_id', 'track_album_name'],
    as_index=False
)['track_popularity'].mean()

# Mengurutkan album dari yang paling populer ke yang kurang populer untuk setiap artis
top_albums = album_popularity.sort_values(
    ['track_artist', 'track_popularity'],
    ascending=[True, False]
)

# Mengambil satu album teratas (paling populer) dari setiap artis
top_album_per_artist = top_albums.drop_duplicates('track_artist', keep='first')

# Menampilkan 10 album teratas dari hasil tersebut
print(top_album_per_artist.head(10))
```
   Insight: Diperoleh daftar album paling populer dari setiap artis, yang dapat digunakan untuk analisis lebih lanjut seperti rekomendasi atau pengelompokan.

5. Menyiapkan Fitur Numerik untuk Model
   Beberapa fitur numerik dipilih dari dataset untuk dianalisis dan digunakan dalam model. Duplikasi berdasarkan track_name juga dihapus untuk menjaga keunikan tiap lagu.
```python
# Fungsi untuk mengekstrak fitur numerik dari lagu untuk keperluan analisis
def prepare_features(df, text_col='track_name', drop_dupes=True):
    # Daftar fitur numerik yang digunakan
    selected_features = [
        'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'track_popularity'
    ]

    # Menyalin kolom fitur dan kolom nama lagu
    data = df[[*selected_features, text_col]].copy()

    # Opsi: hapus duplikat berdasarkan nama lagu
    if drop_dupes:
        data.drop_duplicates(subset=text_col, inplace=True)

    # Mengatur nama lagu sebagai index
    data.set_index(text_col, inplace=True)

    # Mengembalikan DataFrame fitur dan nama-nama fiturnya
    return data, selected_features

# Menjalankan fungsi dan menyimpan hasilnya
temp, features = prepare_features(df_song)
```
   Tujuan: Menyiapkan fitur-fitur numerik yang relevan

6. Normalisasi Data Numerik
   Fitur numerik yang telah dipilih dinormalisasi menggunakan Min-Max Scaling agar berada dalam rentang 0 hingga 1.
```python
scaler = MinMaxScaler()
scaled_numeric = scaler.fit_transform(temp)
```

## Modeling
Pada tahapan ini, saya membangun sistem rekomendasi lagu berbasis content-based filtering dengan pendekatan kombinasi antara fitur numerik dan representasi teks menggunakan TF-IDF. Sistem rekomendasi ini bertujuan memberikan saran lagu yang mirip dengan lagu yang dipilih pengguna berdasarkan karakteristik akustik dan nama lagu.

1. Representasi Teks Menggunakan TF-IDF
   Langkah pertama adalah mengubah fitur teks track_name menjadi vektor numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency). Teknik ini membantu menangkap kata-kata yang unik dari nama lagu untuk memahami konteksnya secara semantik. Proses dilakukan sebagai berikut:
```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(temp.index)
```
   TF-IDF membantu membedakan lagu yang memiliki nama serupa atau mengandung kata kunci spesifik yang bisa diasosiasikan dengan genre atau tema tertentu.

 2. Menggabungkan Fitur Numerik dan Teks
    Setelah mendapatkan representasi teks dan fitur numerik yang sudah dinormalisasi, kedua jenis fitur tersebut digabungkan untuk menghasilkan satu vektor representasi untuk setiap lagu:
```python
combined_features = hstack([scaled_numeric, tfidf_matrix])
```
   Dengan menggabungkan kedua jenis fitur ini, sistem rekomendasi dapat mempertimbangkan baik karakteristik audio lagu (seperti danceability, tempo, valence) maupun informasi semantik dari nama lagu.

3. Perhitungan Kemiripan antar Lagu
   Kemiripan antar lagu dihitung menggunakan cosine similarity terhadap matriks gabungan fitur. Semakin tinggi nilai cosine similarity antara dua lagu, semakin besar kemungkinan keduanya mirip:
```python
sim_df = pd.DataFrame(
    cosine_similarity(combined_features),
    index=temp.index,
    columns=temp.index
)
```
Hasil dari proses ini adalah matriks kemiripan (similarity matrix) yang dapat digunakan untuk merekomendasikan lagu-lagu yang paling mirip dengan lagu input pengguna.


## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
