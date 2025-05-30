# Laporan Proyek Machine Learning - Daffa Aprilian Herdikaputra

## Domain Proyek

Dalam beberapa tahun terakhir, pasar properti di Indonesia mengalami pertumbuhan yang signifikan dengan fluktuasi harga yang cukup dinamis. Prediksi harga rumah menjadi hal yang sangat penting bagi berbagai pihak seperti pembeli potensial, penjual, agen real estate, dan investor [1]. Dengan perkembangan teknologi, pendekatan machine learning untuk memprediksi harga rumah menawarkan solusi yang lebih akurat dibandingkan metode tradisional.

Prediksi harga rumah perlu diselesaikan karena:
1. Membantu pembeli dalam mengambil keputusan pembelian yang optimal
2. Membantu penjual dan agen properti dalam menentukan harga yang kompetitif
3. Memberikan wawasan bagi pengembang properti untuk strategi pembangunan dan pemasaran
4. Mendukung institusi keuangan dalam proses penilaian properti untuk keperluan kredit

Berdasarkan riset yang dilakukan oleh Bank Indonesia, faktor-faktor seperti lokasi, luas tanah, luas bangunan, dan fasilitas menjadi penentu utama harga properti residensial [2]. Oleh karena itu, pemodelan prediktif dengan mempertimbangkan variabel-variabel tersebut menjadi sangat relevan untuk menghasilkan estimasi harga yang akurat.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, berikut adalah rumusan masalah yang akan diselesaikan dalam proyek ini:
- Bagaimana cara memprediksi harga rumah secara akurat berdasarkan fitur-fitur yang tersedia?
- Fitur apa saja yang paling berpengaruh dalam menentukan harga rumah?
- Model machine learning apa yang memberikan hasil prediksi paling akurat untuk kasus ini?

### Goals
Tujuan dari proyek ini adalah:
- Mengembangkan model machine learning yang dapat memprediksi harga rumah dengan tingkat akurasi yang tinggi
- Mengidentifikasi dan menganalisis fitur-fitur yang memiliki pengaruh signifikan terhadap harga rumah
- Membandingkan performa beberapa algoritma machine learning untuk menemukan model terbaik dalam prediksi harga rumah

### Solution Statements
Untuk mencapai tujuan di atas, berikut adalah solusi yang akan diterapkan:
- Mengimplementasikan tiga algoritma machine learning untuk prediksi harga rumah: K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost
- Melakukan evaluasi terhadap ketiga model menggunakan metrik Mean Squared Error (MSE) untuk menentukan model dengan performa terbaik
- Menggunakan data yang telah dipreparasi dengan teknik standarisasi dan penanganan outlier untuk meningkatkan akurasi model

## Data Understanding

### Informasi Dataset
Dataset yang digunakan dalam proyek ini adalah data harga rumah yang berisi informasi tentang berbagai properti residensial. Dataset ini memiliki spesifikasi sebagai berikut:

### Jumlah Data dan Kolom:
Dataset awal: 1.010 baris dan 8 kolom

### Kondisi Data:
Missing values: Tidak ditemukan missing value pada dataset
Outliers: Ditemukan outliers pada beberapa fitur numerik yang telah ditangani menggunakan metode IQR
Duplikasi data: Tidak ditemukan data duplikat

Sumber Dataset
Dataset ini diperoleh dari Kaggle "Daftar harga rumah" yang berisi informasi harga rumah di daerah Tebet dengan berbagai karakteristik fisik dan harga jual yang tercatat.

Link: https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah

### Variabel-variabel pada dataset Harga Rumah adalah sebagai berikut:
- **HARGA**: Harga jual rumah (dalam satuan tertentu)
- **LT** : Luas tanah properti
- **LB** : Luas bangunan properti
- **KT** : Jumlah kamar tidur
- **KM** : Jumlah kamar mandi
- **GRS**: Jumlah kapasitas mobil dalam garasi

Catatan: Kolom 'NO' dan 'NAMA RUMAH' dihapus karena tidak memberikan informasi yang relevan untuk prediksi harga.

### Analisis Eksplorasi Data
1. **Statistik Deskriptif**:
   Analisis statistik deskriptif menunjukkan bahwa dataset memiliki variasi yang cukup tinggi dalam harga rumah, dengan nilai minimum, maksimum, rata-rata, dan standar deviasi yang memberikan gambaran tentang distribusi data.

2. **Analisis Missing Value**:
   Pemeriksaan terhadap missing value dilakukan untuk memastikan kualitas data. Berdasarkan hasil analisis, tidak ditemukan missing value pada dataset yang digunakan.

3. **Analisis Outlier**:
   Visualisasi dengan boxplot menunjukkan adanya beberapa outlier pada variabel LT, LB, KT, KM, GRS. Outlier dapat mempengaruhi performa model sehingga perlu ditangani.

4. **Univariate Analysis**:
   Analisis univariat melalui histogram dilakukan untuk memahami distribusi masing-masing variabel. Hasil menunjukkan bahwa beberapa variabel, seperti HARGA, memiliki distribusi yang cenderung miring ke kanan (right-skewed).

5. **Multivariate Analysis**:
   Analisis multivariat dilakukan untuk memahami hubungan antar variabel. Hasil pairplot dan heatmap korelasi menunjukkan bahwa terdapat korelasi positif yang kuat antara luas tanah (LT) dan harga rumah (HARGA) dengan nilai korelasi 0.8, diikuti oleh korelasi antara luas bangunan (LB) dan harga rumah sebesar 0.68. Ini mengindikasikan bahwa semakin besar luas tanah dan luas bangunan, semakin tinggi harga rumah, dengan luas tanah menjadi faktor yang lebih berpengaruh.

## Data Preparation

Beberapa teknik data preparation diterapkan untuk mempersiapkan data agar optimal untuk pemodelan:

1. **Penghapusan Fitur yang Tidak Relevan**:
   Kolom 'NO' dan 'NAMA RUMAH' dihapus karena tidak memberikan informasi yang signifikan untuk prediksi harga rumah. Fitur-fitur ini bersifat identifikasi dan tidak memiliki korelasi dengan harga.

2. **Penanganan Outlier**:
   Outlier pada fitur numerik ditangani dengan metode IQR (Interquartile Range). Baris data yang mengandung nilai di luar rentang Q1 - 1.5*IQR dan Q3 + 1.5*IQR dihapus untuk menghindari bias pada model.

3. **Train-Test Split**:
   Dataset dibagi menjadi data latih (80%) dan data uji (20%) dengan random_state=111 untuk memastikan reproduktibilitas hasil. Pembagian ini penting untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya.

4. **Standarisasi Fitur Numerik**:
   Fitur numerik (LT, LB, KT, KM, GRS) distandarisasi menggunakan StandardScaler untuk membuat nilai rata-rata=0 dan standar deviasi=1. Standarisasi diperlukan terutama untuk algoritma KNN yang sensitif terhadap skala data.

Proses data preparation ini penting untuk:
- Meningkatkan kualitas data dengan menghilangkan noise
- Menyeragamkan skala fitur untuk algoritma yang sensitif terhadap skala
- Mencegah overfitting akibat outlier
- Memungkinkan evaluasi yang adil terhadap model

### Jumlah Data dan Kolom
Jumlah data setelah data praparation: 695 baris dan 6 kolom

## Modeling

Pada tahap ini, tiga algoritma machine learning diimplementasikan untuk memprediksi harga rumah:

### Model 1: K-Nearest Neighbors (KNN)

**Pembahasan Cara Kerja:**
K-Nearest Neighbors (KNN) adalah algoritma supervised learning yang melakukan prediksi berdasarkan nilai k tetangga terdekat dalam ruang fitur. Algoritma ini bekerja dengan menghitung jarak euclidean antara data uji dengan seluruh data latih, kemudian mengambil k data terdekat untuk menentukan prediksi. Untuk masalah regresi, KNN mengembalikan rata-rata nilai target dari k tetangga terdekat.

**Pembahasan Parameter:**
- `n_neighbors=3`: Menggunakan 3 tetangga terdekat untuk prediksi. Nilai k yang kecil membuat model lebih sensitif terhadap noise, sedangkan nilai k yang besar membuat model lebih stabil tetapi kurang detail.

**Kelebihan/Kekurangan:**
Kelebihan:
- Sederhana dan mudah diinterpretasi
- Tidak membutuhkan asumsi tentang bentuk data
- Efektif untuk dataset kecil hingga menengah
- Tidak perlu training time yang lama

Kekurangan:
- Sensitif terhadap skala fitur (memerlukan normalisasi)
- Performa dapat menurun pada dataset berdimensi tinggi (curse of dimensionality)
- Komputasi berat saat prediksi karena harus menghitung jarak dengan semua data training
- Sensitif terhadap outlier

### Model 2: Random Forest

**Pembahasan Cara Kerja:**
Random Forest adalah ensemble method yang menggunakan banyak decision tree untuk melakukan prediksi. Setiap tree dilatih pada subset data yang berbeda (bootstrap sampling) dan menggunakan subset fitur yang dipilih secara acak pada setiap split. Prediksi akhir merupakan rata-rata dari prediksi semua tree untuk masalah regresi.

**Pembahasan Parameter:**
- `n_estimators=50`: Menggunakan 50 pohon keputusan. Semakin banyak tree, semakin stabil modelnya tetapi membutuhkan lebih banyak waktu komputasi.
- `max_depth=16`: Kedalaman maksimum setiap pohon adalah 16. Parameter ini mengontrol kompleksitas model dan mencegah overfitting.
- `random_state=111`: Menjamin reproduktibilitas hasil
- `n_jobs=-1`: Menggunakan semua prosesor yang tersedia untuk mempercepat training

**Kelebihan/Kekurangan:**
Kelebihan:
- Tahan terhadap overfitting karena menggunakan ensemble
- Dapat menangani hubungan non-linear dengan baik
- Memberikan informasi feature importance
- Robust terhadap outlier
- Dapat menangani missing value dengan baik

Kekurangan:
- Lebih kompleks dibandingkan model sederhana
- Membutuhkan lebih banyak memori dan waktu komputasi
- Interpretabilitas lebih rendah dibanding single decision tree
- Dapat overfitting pada dataset dengan noise tinggi

### Model 3: AdaBoost

**Pembahasan Cara Kerja:**
AdaBoost (Adaptive Boosting) adalah algoritma boosting yang membangun model secara sequential. Setiap model weak learner (biasanya decision stump) dilatih untuk memperbaiki kesalahan model sebelumnya. Data yang salah diprediksi diberi bobot lebih tinggi pada iterasi berikutnya, sehingga model fokus memperbaiki kesalahan tersebut.

**Pembahasan Parameter:**
- `learning_rate=0.05`: Tingkat pembelajaran yang rendah untuk menghindari overfitting. Learning rate mengontrol kontribusi setiap weak learner terhadap prediksi final.
- `random_state=111`: Menjamin reproduktibilitas hasil

**Kelebihan/Kekurangan:**
Kelebihan:
- Secara bertahap memperbaiki kesalahan model dasar
- Jarang mengalami overfitting jika parameter diset dengan baik
- Secara otomatis melakukan feature selection
- Dapat mencapai performa tinggi dengan model dasar yang sederhana

Kekurangan:
- Sensitif terhadap noise dan outlier
- Performa sangat tergantung pada kualitas weak learner
- Dapat lebih lambat untuk dilatih dibandingkan Random Forest
- Rentan terhadap overfitting jika learning rate terlalu tinggi

## Evaluation

Pada proyek ini, metrik evaluasi utama yang digunakan adalah Mean Squared Error (MSE). MSE dipilih karena:
1. Sesuai untuk masalah regresi seperti prediksi harga rumah
2. Memberikan penalti lebih besar untuk kesalahan besar, yang penting dalam konteks prediksi harga
3. Memiliki interpretasi yang jelas dalam unit yang sama dengan target (setelah diakarkan)

### Formula MSE:
MSE = (1/n) Σ(y_true - y_pred)²

Dimana:
- n adalah jumlah sampel
- y_true adalah nilai sebenarnya
- y_pred adalah nilai prediksi

### Hasil Evaluasi

Catatan: Untuk memudahkan pembacaan, semua nilai MSE yang ditampilkan dalam laporan ini telah dibagi dengan faktor 1.000 (x1000).

| Model     | MSE Train | MSE Test |
|-----------|-----------|----------|
|KNN	      |1729.50    |2876.96   |
|RF	      |519.19     |2425.02   | 
|Boosting	|2344.43    |2425.22   |

*Nilai MSE asli (sebelum dibagi 1000):*
- KNN: Train = 1,729,495,362,316,779.5, Test = 2,876,964,775,824,014.0
- RF: Train = 519,186,521,428,438.5625, Test = 2,425,023,465,318,333.5
- Boosting: Train = 2,344,413,129,177,157.5, Test = 2,425,221,097,144,163.0

### Analisis Hasil Evaluasi

Berdasarkan hasil evaluasi di atas, dapat diamati bahwa:

1. **Random Forest** menunjukkan performa terbaik dengan MSE test terendah (2425.02), yang menunjukkan kemampuan generalisasi yang baik pada data yang belum pernah dilihat.

2. **AdaBoost** memiliki MSE test yang hampir sama dengan Random Forest (2425.22), menunjukkan performa yang kompetitif.

3. **KNN** memiliki MSE test tertinggi (2876.96), namun menariknya model ini menunjukkan MSE train yang cukup tinggi (1729.50), mengindikasikan bahwa model mungkin underfitting atau parameter k=3 kurang optimal.

4. Semua model menunjukkan peningkatan MSE dari data train ke data test, yang merupakan hal normal dan mengindikasikan adanya sedikit overfitting, namun masih dalam batas yang wajar.

### Evaluasi dengan Sampel Prediksi

Sebagai evaluasi tambahan, dilakukan prediksi pada sampel pertama data test dengan nilai aktual (y_true) = 6,100,000,000:

| Model         | Prediksi      | Akurasi (MAPE) |
|---------------|---------------|----------------|
| KNN           | 6,116,667,000 | 99.73%         |
| Random Forest | 6,433,735,000 | 94.53%         |
| AdaBoost      | 7,279,615,000 | 80.66%         |

Hasil evaluasi sampel individual menunjukkan bahwa meskipun Random Forest memiliki MSE keseluruhan terbaik, KNN memberikan prediksi paling akurat untuk sampel spesifik ini dengan akurasi 99.73%. Hal ini menunjukkan bahwa KNN dapat sangat akurat untuk kasus-kasus tertentu yang memiliki tetangga terdekat dengan karakteristik serupa dalam data training.

### Hubungan dengan Business Understanding

Hasil evaluasi ini menjawab problem statement yang diajukan:

1. **Cara memprediksi harga rumah secara akurat**: Model Random Forest memberikan prediksi paling akurat dengan MSE terendah, menunjukkan bahwa pendekatan ensemble learning efektif untuk prediksi harga rumah.

2. **Fitur yang berpengaruh**: Berdasarkan analisis korelasi sebelumnya, luas tanah (LT) dan luas bangunan (LB) menjadi fitur paling berpengaruh terhadap harga rumah.

3. **Model terbaik**: Random Forest memberikan hasil prediksi paling akurat dan konsisten, menjadikannya pilihan terbaik untuk sistem prediksi harga rumah.

### Kesimpulan Model Terbaik

Berdasarkan hasil evaluasi komprehensif, dapat disimpulkan bahwa:

**Untuk Prediksi Umum (Berdasarkan MSE):**
**Random Forest** direkomendasikan sebagai model utama karena:
- Memiliki MSE test terendah (2425.02)
- Menunjukkan kemampuan generalisasi yang baik dan konsisten
- Robust terhadap outlier dan noise
- Memberikan keseimbangan yang baik antara akurasi training dan testing

**Untuk Kasus Spesifik:**
**KNN** menunjukkan potensi akurasi tinggi (99.73%) pada sampel individual, yang mengindikasikan bahwa model ini efektif ketika data test memiliki karakteristik yang sangat mirip dengan data training.

**Rekomendasi Implementasi:**
Untuk sistem prediksi harga rumah yang praktis, Random Forest tetap menjadi pilihan utama karena memberikan performa yang stabil dan dapat diandalkan secara keseluruhan. Namun, KNN dapat dipertimbangkan sebagai model pendukung untuk kasus-kasus khusus dimana diperlukan akurasi tinggi untuk properti dengan karakteristik yang sangat spesifik.

Model ini berhasil mencapai goal yang ditetapkan yaitu mengembangkan model machine learning dengan tingkat akurasi tinggi untuk prediksi harga rumah.

## Referensi

[1] S. Mullainathan and J. Spiess, "Machine Learning: An Applied Econometric Approach," Journal of Economic Perspectives, vol. 31, no. 2, pp. 87-106, 2017.

[2] Bank Indonesia, "Perkembangan Properti Residensial di Indonesia," Survei Harga Properti Residensial, 2023.