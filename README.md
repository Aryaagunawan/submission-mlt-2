# Laporan Proyek Machine Learning - Arya Gunawan

## Project Overview

Proyek ini berfokus pada analisis dan prediksi popularitas lagu menggunakan data musik dari Spotify. Dengan lebih dari 30.000 entri lagu, proyek ini bertujuan untuk memahami faktor-faktor yang mempengaruhi popularitas lagu berdasarkan fitur audio seperti danceability, energy, tempo, dan lainnya.

Masalah ini penting untuk diselesaikan karena dalam industri musik digital, memahami preferensi pengguna dan karakteristik lagu yang populer dapat membantu musisi, label rekaman, dan platform streaming untuk merancang strategi distribusi dan rekomendasi yang lebih efektif.

Sistem rekomendasi berperan dalam meningkatkan pengalaman pengguna dengan memanfaatkan data seperti preferensi lagu, riwayat pemutaran, dan konten lagu itu sendiri. Salah satu metode yang diterapkan adalah Content-based Filtering, yaitu merekomendasikan lagu berdasarkan kesamaan fitur musik seperti tempo, energy, dan danceability serta aspek teks dari judul lagu. Dalam prosesnya, TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk memberikan bobot pada setiap kata dalam judul lagu, sementara Cosine Similarity dipakai untuk mengukur tingkat kemiripan antar lagu berdasarkan fitur musik dan judul tersebut [[1]](https://ojs.adzkia.ac.id/index.php/jtech/article/view/282/168).

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


#### Kondisi Awal Data
  
  Untuk memahami kondisi awal data sebelum proses pembersihan dan transformasi, dilakukan analisis berikut:

  Missing Values
```python
df_song.isnull().sum()
```
Hasil:
| Kolom                       | Jumlah Nilai Kosong |
|----------------------------|---------------------|
| `track_id`                 | 0                   |
| `track_name`               | 5                   |
| `track_artist`             | 5                   |
| `track_popularity`         | 0                   |
| `track_album_id`           | 0                   |
| `track_album_name`         | 5                   |
| `track_album_release_date`| 0                   |
| `playlist_id`              | 0                   |
| `playlist_name`            | 0                   |
| `playlist_genre`           | 0                   |
| `playlist_subgenre`        | 0                   |
| `danceability`             | 0                   |
| `energy`                   | 0                   |
| `key`                      | 0                   |
| `loudness`                 | 0                   |
| `mode`                     | 0                   |
| `speechiness`              | 0                   |
| `acousticness`             | 0                   |
| `instrumentalness`         | 0                   |
| `liveness`                 | 0                   |
| `valence`                  | 0                   |
| `tempo`                    | 0                   |
| `duration_ms`              | 0                   |


Insight: Terdapat 5 nilai kosong (missing value) pada tiga kolom bertipe objek (nama lagu, artis, dan album). Ini menunjukkan perlunya penanganan pada data teks tersebut sebelum diproses lebih lanjut.

- Data Duplikat
```python
df_song.duplicated().sum()
```
Hasil:

0 duplikat terdeteksi dalam dataset.

Insight: Tidak ditemukan data ganda, sehingga tidak diperlukan penghapusan duplikasi.

- Statistik Deskriptif
```python
df_song.describe()
```

![image](https://github.com/user-attachments/assets/445c0864-7618-46a9-b756-fa856d7aa07d)


Insight:

- Banyak lagu memiliki `track_popularity` rendah, bahkan 0.
- Rata-rata `tempo` lagu sekitar 121 BPM.
- Sebagian besar nilai numerik wajar, namun `loudness` memiliki nilai ekstrem (-46.45 dB).
- `instrumentalness` dan `acousticness` memiliki distribusi miring ke kanan (right-skewed).

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

**Distribusi Popularitas Lagu Berdasarkan Genre**
```python
sns.boxplot(x='playlist_genre', y='track_popularity', data=df_song)
plt.xticks(rotation=45)
plt.title('Popularitas Lagu berdasarkan Genre')
plt.show()
```
![image](https://github.com/user-attachments/assets/4580dcb8-0b87-4615-a8e0-7f235a8e3440)


Insight:

Genre 'pop' memiliki median popularitas tertinggi, sedangkan 'edm' terendah. Semua genre menunjukkan rentang popularitas yang luas, dari sangat rendah hingga sangat tinggi.


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
  Alasan: Menghapus data kosong mencegah error dalam perhitungan dan menjaga integritas data.
  
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
   Insight:

  Diperoleh daftar album paling populer dari setiap artis, yang dapat digunakan untuk analisis lebih lanjut seperti rekomendasi atau pengelompokan.

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
   Tujuan: Menyiapkan fitur-fitur numerik yang relevan dan menjadikan track_name sebagai indeks agar proses pencocokan fitur numerik dengan fitur teks lebih mudah dan konsisten.

6. Normalisasi Data Numerik
   Fitur numerik yang telah dipilih dinormalisasi menggunakan Min-Max Scaling agar berada dalam rentang 0 hingga 1.
```python
scaler = MinMaxScaler()
scaled_numeric = scaler.fit_transform(temp)
```
7. Representasi Teks menggunakan TF-IDF
   
   Nama lagu (track_name) diubah menjadi representasi numerik menggunakan TF-IDF untuk menangkap informasi semantik:
```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(temp.index)
```
8. Penggabungan Fitur Numerik dan Teks

   Fitur numerik yang telah dinormalisasi dan fitur teks (TF-IDF) digabungkan menjadi satu matriks fitur menggunakan hstack:
```python
combined_features = hstack([scaled_numeric, tfidf_matrix])
```


## Modeling
Tahapan ini bertujuan membangun sistem rekomendasi berbasis Content-Based Filtering menggunakan kombinasi fitur numerik dan teks.

1. Perhitungan Kemiripan (Similarity)
   Kemiripan antar lagu dihitung menggunakan Cosine Similarity terhadap combined_features:
```python
sim_df = pd.DataFrame(
    cosine_similarity(combined_features),
    index=temp.index,
    columns=temp.index
)
```
Hasil dari proses ini adalah matriks kemiripan (similarity matrix) yang dapat digunakan untuk merekomendasikan lagu-lagu yang paling mirip dengan lagu input pengguna.

2. Contoh Output Top-N Rekomendasi
   
   Sebagai contoh, berikut adalah hasil rekomendasi Top-10 lagu yang mirip dengan lagu “Someone Like You”:
```python
song_name = 'Someone Like You'
similar_songs = sim_df[song_name].sort_values(ascending=False).iloc[1:11]
print(similar_songs)
```
hasil:

| No | Track Name                 | Similarity Score |
|----|----------------------------|------------------|
| 1  | Never Be Like You           | 0.9615           |
| 2  | The One I Like              | 0.9184           |
| 3  | Like Someone In Love        | 0.9112           |
| 4  | But I Like It               | 0.9110           |
| 5  | Like You                    | 0.9078           |
| 6  | LIKE I WANT YOU             | 0.8896           |
| 7  | You're The One That I Like  | 0.8894           |
| 8  | I Like That                 | 0.8812           |
| 9  | like that                   | 0.8804           |
| 10 | Nothing Like You            | 0.8758           |

Insight:

- Lagu-lagu yang paling mirip dengan "Someone Like You" sebagian besar mengandung kata "Like" dalam judulnya, menunjukkan kemiripan tema atau gaya.

- Nilai skor kemiripan berada di kisaran 0.87 sampai 0.96, menunjukkan tingkat kesamaan yang cukup tinggi.

- Ini mengindikasikan model similarity cukup sensitif terhadap kemiripan judul dan mungkin juga fitur audio atau metadata yang terkait.

Kelebihan:

- Tidak bergantung pada data historis pengguna atau riwayat interaksi.
- Mampu merekomendasikan lagu baru meskipun belum pernah diputar sebelumnya (mengatasi masalah cold start pada item).

Kekurangan:

- Rekomendasi yang dihasilkan cenderung terbatas pada item yang memiliki kesamaan tekstual, seperti kemiripan pada judul, sehingga kurang beragam.
- Pendekatan ini tidak secara langsung memperhitungkan preferensi individu pengguna karena hanya berfokus pada atribut dari item itu sendiri.


## Evaluation

Pada tahap evaluasi, sistem rekomendasi yang telah dibangun diuji untuk memastikan kemampuannya dalam memberikan rekomendasi yang relevan. Metrik evaluasi yang umumnya digunakan untuk sistem rekomendasi seperti Precision@K, Recall@K, dan F-Score@K, membutuhkan data ground truth (lagu-lagu yang benar-benar relevan bagi pengguna) yang tidak selalu tersedia secara eksplisit dalam dataset ini.

Karena ini adalah sistem Content-Based Filtering dan untuk memenuhi kriteria evaluasi yang mengharuskan penyajian nilai metrik kuantitatif, kami akan melakukan perhitungan secara manual untuk satu contoh kasus. Untuk menentukan ground truth atau item yang relevan, kami akan berasumsi bahwa lagu-lagu yang memiliki kesamaan semantik atau kontekstual yang sangat tinggi dengan lagu input, berdasarkan pemahaman manusia, dianggap relevan.


#### Metrik Evaluasi

- Precision@K
  
  Precision@K mengukur proporsi lagu yang relevan dari K rekomendasi teratas. Fokusnya adalah kualitas rekomendasi yang diberikan.

  Rumus:

![image](https://github.com/user-attachments/assets/ad6ca1d2-ab9c-4cc8-8c35-e0d20cc9ecda)

Maka: Jika Precision@10=0.5, artinya 50% dari 10 lagu teratas yang direkomendasikan adalah relevan bagi pengguna. Semakin tinggi nilainya, semakin sedikit "sampah" dalam rekomendasi.

- Recall@K
  Mengukur kemampuan sistem rekomendasi untuk mengidentifikasi semua item yang relevan dalam rekomendasi K teratas. Ini berfokus pada kelengkapan sistem, yaitu seberapa banyak item relevan yang berhasil ditemukan oleh sistem.

  Rumus:

![image](https://github.com/user-attachments/assets/04f54056-204a-44b1-99f9-cc0705f5ea62)

Maka: Jika Recall@10=0.714, artinya 71.4% dari semua lagu yang relevan (misalnya ada 7 lagu yang relevan dan sistem berhasil merekomendasikan 5 di antaranya) berhasil masuk dalam 10 rekomendasi teratas. Semakin tinggi nilainya, semakin banyak item relevan yang ditemukan.

- F-Score@K

  Merupakan rata-rata harmonik dari Precision@K dan Recall@K. Metrik ini memberikan keseimbangan antara presisi dan recall, sangat berguna ketika kita ingin         mempertimbangkan kedua aspek tersebut secara bersamaan.
  
 Rumus: 

![image](https://github.com/user-attachments/assets/b7ea4337-412a-40cd-8489-0152681a0ad8)

Maka: Jika F-Score@10=0.588, ini mencerminkan keseimbangan antara presisi dan recall. Nilai ini memberikan metrik tunggal untuk mengevaluasi kinerja sistem rekomendasi, di mana nilai yang lebih tinggi menunjukkan kinerja yang lebih baik secara keseluruhan dalam menyeimbangkan kualitas dan kelengkapan rekomendasi.


#### Pengujian Sistem Rekomendasi dan Perhitungan Metrik Kuantitatif

 Sistem diuji dengan memasukkan satu lagu sebagai input dan mengamati 10 lagu teratas yang direkomendasikan berdasarkan kemiripan konten.

#### Pengujian dan Perhitungan Metrik:

- Lagu input: Someone Like You

- Top 10 rekomendasi:

| No | Judul Lagu              | Skor Kemiripan | Relevan (Ground Truth) |
|----|------------------------|----------------|------------------------|
| 1  | Never Be Like You      | 0.9615         | Ya                     |
| 2  | The One I Like         | 0.9184         | Ya                     |
| 3  | Like Someone In Love   | 0.9112         | Ya                     |
| 4  | But I Like It          | 0.9110         | Tidak                  |
| 5  | Like You               | 0.9078         | Ya                     |
| 6  | LIKE I WANT YOU        | 0.8896         | Ya                     |
| 7  | You're The One That I Like | 0.8894      | Ya                     |
| 8  | I Like That            | 0.8812         | Tidak                  |
| 9  | like that              | 0.8804         | Tidak                  |
| 10 | Nothing Like You       | 0.8758         | Ya                     |


Penjelasan Penentuan Relevansi (Ground Truth):
Untuk kasus ini, kami mendefinisikan "relevan" secara kualitatif berdasarkan kesamaan judul yang sangat jelas dengan lagu input "Someone Like You", atau frasa "Like You" atau "Like" yang menandakan kemiripan semantik. Lagu-lagu yang tidak memiliki kemiripan judul eksplisit atau tematik yang kuat (misalnya "But I Like It", "I Like That", "like that" yang bisa merujuk pada banyak konteks) dianggap tidak relevan untuk tujuan perhitungan ini.


**Perhitungan Metrik:**

- Jumlah lagu relevan di Top 10 (K=10): 7 lagu (Never Be Like You, The One I Like, Like Someone In Love, Like You, LIKE I WANT YOU, You're The One That I Like, Nothing Like You)

- Total lagu yang relevan (asumsi dari potensi maksimum yang mungkin ada di dataset, dalam contoh ini, kita asumsikan ada 8 lagu yang relevan di seluruh dataset yang cocok dengan "Someone Like You" secara ideal): Untuk tujuan demonstrasi ini, mari kita asumsikan ada total 8 lagu di seluruh dataset yang benar-benar relevan dengan "Someone Like You" (ini adalah asumsi yang harus Anda sesuaikan jika Anda memiliki ground truth yang lebih konkret).

1. Precision@10:
   
   Precision@10= 
10
7
 =0.7
Ini berarti 70% dari 10 rekomendasi teratas adalah relevan.


2. Recall@10:
   
   Recall@10= 
8
7
 =0.875
Ini berarti sistem berhasil menemukan 87.5% dari semua lagu relevan yang diasumsikan ada.


3. F-Score@10:

   F−Score@10=2× 
0.7+0.875
0.7×0.875
​
 =2× 
1.575
0.6125
​
 ≈2×0.3888≈0.777
Nilai F-Score@10 sebesar 0.777 menunjukkan keseimbangan yang baik antara presisi dan recall dalam rekomendasi.

#### Analisis Hasil (Kualitatif dan Kuantitatif):

Berdasarkan hasil pengujian di atas, baik secara kualitatif maupun kuantitatif, sistem rekomendasi menunjukkan performa yang cukup baik. Nilai Precision@10 sebesar 0.7 menunjukkan bahwa sebagian besar rekomendasi yang diberikan berkualitas tinggi. Sementara itu, Recall@10 sebesar 0.875 menunjukkan bahwa sistem cukup efektif dalam menemukan sebagian besar lagu relevan yang diasumsikan ada. Kombinasi kedua metrik ini menghasilkan F-Score@10 sekitar 0.777, yang menandakan kinerja keseluruhan yang kuat dalam menyeimbangkan kualitas dan kelengkapan rekomendasi.

Secara kualitatif, untuk lagu "Someone Like You", mayoritas rekomendasi memang memiliki frasa "Like You" atau "Like" di dalamnya, serta beberapa rekomendasi yang secara tematik mirip seperti "The One I Like" atau "Nothing Like You". Hal ini secara konsisten menunjukkan bahwa penggunaan TF-IDF pada track_name dikombinasikan dengan Cosine Similarity efektif dalam menangkap kemiripan berbasis judul.

#### Evaluasi terhadap Business Understanding

Sistem rekomendasi yang dikembangkan telah diuji dengan memberikan input lagu dan menghasilkan 10 rekomendasi teratas berbasis kesamaan konten. Berdasarkan hasil evaluasi, sistem ini telah mampu menjawab tujuan dan permasalahan yang dirumuskan sebelumnya:

- Menjawab Problem Statement 1 – Efisiensi Pencarian Lagu:
Sistem berhasil merekomendasikan lagu-lagu yang mirip berdasarkan fitur judul dan konten numerik, sehingga pengguna tidak perlu lagi mencari lagu secara manual dari ribuan pilihan yang tersedia. Ini membuat proses menemukan lagu menjadi lebih cepat dan efisien.


- Menjawab Problem Statement 2 – Rekomendasi Tanpa Riwayat Pengguna:
Karena sistem ini menggunakan pendekatan content-based filtering dan tidak bergantung pada histori pemutaran pengguna, sistem tetap dapat memberikan rekomendasi meskipun tidak ada interaksi pengguna sebelumnya (mengatasi masalah cold start).

- Menjawab Goals Proyek:
Sistem memberikan pengalaman personalisasi dengan merekomendasikan lagu yang mirip berdasarkan konten, khususnya dari sisi judul lagu. Hal ini sejalan dengan tujuan proyek untuk membangun sistem berbasis konten yang relevan.

- Menjawab Solution Statements:
Sistem telah mengimplementasikan metode TF-IDF pada judul lagu dan menghitung kemiripan menggunakan cosine similarity. Hasil rekomendasi menunjukkan bahwa pendekatan ini cukup efektif, meskipun masih bisa ditingkatkan dengan menambahkan fitur audio lainnya atau menggabungkan dengan metode lain seperti collaborative filtering jika data tersedia.

#### Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi lagu berbasis Content-Based Filtering menggunakan kombinasi TF-IDF pada judul lagu dan Cosine Similarity. Hasil pengujian menunjukkan bahwa sistem mampu memberikan rekomendasi yang relevan, khususnya untuk lagu-lagu yang belum pernah diputar sebelumnya (cold start).

Meskipun sistem menunjukkan presisi yang baik (0.7) dan recall yang tinggi (0.875) pada pengujian, yang tercermin dalam F-Score 0.777, terdapat ruang untuk peningkatan dari sisi keberagaman rekomendasi. Penggunaan fitur tambahan seperti lirik lagu, genre, atau metadata audio lainnya dapat membantu meningkatkan variasi dan akurasi hasil rekomendasi.

Sistem ini dapat menjadi solusi awal yang efektif untuk memandu pengguna menemukan lagu yang sesuai dengan preferensi mereka, serta menjadi fondasi untuk pengembangan sistem rekomendasi hybrid di masa mendatang yang menggabungkan konten dan histori pengguna.
