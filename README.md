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
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
