## Laporan Proyek Machine Learning - Prediksi Cuaca

### Domain Proyek

#### Latar Belakang

Prediksi cuaca adalah salah satu tantangan penting dalam ilmu data, dengan dampak yang signifikan pada kehidupan manusia. Kemampuan untuk memprediksi kondisi cuaca dengan akurasi tinggi membantu masyarakat mempersiapkan diri terhadap perubahan cuaca, mendukung sektor pertanian, dan meningkatkan keselamatan dalam berbagai aktivitas. Proyek ini menggunakan dataset _Weather Prediction Dataset_ yang berisi data dari European Climate Assessment & Dataset (ECA&D).

#### Sumber Dataset

Dataset ini berasal dari proyek ECA&D yang menyediakan data observasi harian dari berbagai stasiun meteorologi di Eropa dan Mediterania, mencakup 18 lokasi di Eropa dengan rentang waktu dari tahun 2000 hingga 2010. Variabel yang tersedia antara lain suhu rata-rata, suhu maksimum, suhu minimum, kecepatan angin, curah hujan, dan lainnya. Dataset ini telah melalui proses pembersihan untuk menghapus data yang tidak valid (>5%) dan menggantikan nilai hilang (“-9999”) dengan nilai rata-rata.

#### Mengapa Masalah Ini Penting?

- **Keselamatan Publik**: Prediksi cuaca yang akurat dapat mengurangi risiko yang disebabkan oleh cuaca ekstrem.
- **Efisiensi Operasional**: Mendukung perencanaan sektor logistik dan pertanian.
- **Penghematan Biaya**: Mengurangi kerugian finansial akibat ketidaksiapan terhadap perubahan cuaca.

#### Hasil Riset Terkait

1. Klein Tank, A.M.G. et al. (2002). "Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment." _Int. J. of Climatol._, 22, 1441-1453.
2. Rasp, S. et al. (2020). "WeatherBench: A benchmark dataset for data-driven weather forecasting." _Bulletin of the American Meteorological Society_.

---

### Business Understanding

#### Problem Statements

- Bagaimana model machine learning dapat memprediksi variabel cuaca tertentu berdasarkan data historis?
- Algoritma machine learning mana yang memberikan hasil terbaik dalam hal akurasi dan kecepatan prediksi?

#### Goals

- Membangun model machine learning untuk memprediksi kondisi cuaca dengan akurasi tinggi.
- Mengidentifikasi algoritma terbaik berdasarkan kinerja prediksi cuaca.

#### Solution Statements

- Menggunakan model regresi seperti Random Forest dan Gradient Boosting sebagai baseline.
- Menerapkan teknik deep learning seperti Recurrent Neural Networks (RNN) untuk menangkap pola dalam data deret waktu.
- Melakukan hyperparameter tuning untuk meningkatkan performa model, seperti penyesuaian jumlah pohon, learning rate, dan ukuran batch.

---

### Data Understanding

#### Deskripsi Dataset

Dataset ini mencakup data cuaca harian dari 18 lokasi di Eropa selama periode **2000 hingga 2010**. Dataset ini terdiri dari total **3654 baris dan 165 kolom**. Variabel-variabel yang tersedia dalam dataset meliputi:

1. **Suhu**:

   - TG: Suhu rata-rata harian dalam derajat Celsius.
   - TN: Suhu minimum harian dalam derajat Celsius.
   - TX: Suhu maksimum harian dalam derajat Celsius.

2. **Kelembapan (HU)**:

   - Persentase kelembapan udara dalam fraksi 100%.

3. **Kecepatan Angin**:

   - FG: Kecepatan angin rata-rata harian dalam meter per detik (m/s).
   - FX: Kecepatan angin maksimum (gust) dalam meter per detik (m/s).

4. **Tekanan Udara (PP)**:

   - Tekanan udara di permukaan laut dalam satuan 1000 hPa.

5. **Curah Hujan (RR)**:

   - Jumlah curah hujan harian dalam sentimeter (cm).

6. **Durasi Sinar Matahari (SS)**:

   - Durasi penyinaran matahari harian dalam jam.

7. **Atribut Lain**:
   - CC: Tingkat tutupan awan dalam oktas.
   - DD: Arah angin dalam derajat.
   - QQ: Radiasi global dalam satuan 100 W/m².

#### Distribusi Data

Dataset memiliki bentuk deret waktu dengan total **3654 observasi harian** dari berbagai lokasi. Data direpresentasikan dalam kolom-kolom yang mencakup berbagai atribut cuaca seperti suhu, kelembapan, tekanan udara, kecepatan angin, curah hujan, dan lainnya.

#### Kondisi Data

1. **Jumlah Baris dan Kolom**:  
   Dataset memiliki **3654 baris** (observasi) dan **165 kolom** (atribut).

2. **Nilai yang Hilang (Missing Values)**:  
   Dilakukan pengecekan terhadap nilai yang hilang di setiap kolom. Berdasarkan hasil eksplorasi, terdapat beberapa atribut yang memiliki nilai kosong (missing values). Nilai hilang ini telah diatasi menggunakan metode **imputasi**, misalnya pengisian dengan nilai rata-rata atau interpolasi deret waktu.

#### Distribusi Data

Dataset ini diambil dari Kaggle pada tautan berikut:  
[Weather Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/thedevastator/weather-prediction)

#### Uraian Fitur pada Dataset

Berikut adalah uraian singkat seluruh fitur utama yang ada pada dataset:

| **Fitur** | **Deskripsi**                   | **Satuan**      |
| --------- | ------------------------------- | --------------- |
| **DATE**  | Tanggal observasi               | -               |
| **CC**    | Tutupan awan                    | Oktas           |
| **DD**    | Arah angin                      | Derajat         |
| **FG**    | Kecepatan angin rata-rata       | Meter/detik     |
| **FX**    | Kecepatan angin maksimum (gust) | Meter/detik     |
| **HU**    | Kelembapan udara                | % (fraksi 100)  |
| **PP**    | Tekanan udara di permukaan laut | 1000 hPa        |
| **QQ**    | Radiasi global                  | 100 W/m²        |
| **RR**    | Curah hujan                     | Centimeter      |
| **SS**    | Durasi penyinaran matahari      | Jam             |
| **TG**    | Suhu rata-rata                  | Derajat Celsius |
| **TN**    | Suhu minimum                    | Derajat Celsius |
| **TX**    | Suhu maksimum                   |

---

### Data Preparation

#### Tahapan Data Preparation:

1. **Pembersihan Data**:

   - Menghapus kolom dengan lebih dari 5% nilai hilang.
   - Mengganti nilai hilang dengan rata-rata pada kolom yang tersisa.

2. **Transformasi Data**:

   - Mengonversi satuan data untuk meningkatkan keseragaman (misalnya, suhu dalam derajat Celsius, curah hujan dalam cm).
   - Normalisasi data numerik ke rentang [0,1] menggunakan MinMaxScaler.

3. **Pembagian Data**:
   - Dataset dibagi menjadi 70% data pelatihan dan 30% data pengujian.
   - Data pelatihan dibagi lagi menjadi data pelatihan dan validasi dengan rasio 80:20.

---

### Modeling

#### Model yang Digunakan

1. **Random Forest Regressor**:
   **Deskripsi Model:**
   Random Forest Regressor adalah algoritma berbasis ensemble yang menggunakan beberapa pohon keputusan untuk menghasilkan prediksi akhir. Model ini bekerja dengan membuat subset acak dari data pelatihan dan memilih subset fitur secara acak pada setiap split dalam pohon, yang membantu meningkatkan akurasi dan mengurangi overfitting.

**Parameter yang Digunakan:**

- `random_state=42`: Parameter ini digunakan untuk memastikan replikasi hasil eksperimen dengan mengontrol proses pengacakan dalam algoritma.

2. **Gradient Boosting Regressor**:
   **Deskripsi Model:**
   Gradient Boosting Regressor adalah algoritma boosting yang membangun model secara iteratif, di mana setiap model baru mencoba untuk mengurangi kesalahan dari model sebelumnya. Proses ini dilakukan dengan mengoptimalkan fungsi loss menggunakan gradient descent.

**Parameter yang Digunakan:**

- `random_state=42`: Digunakan untuk mengontrol pengacakan dan memastikan hasil yang dapat direplikasi.

**Deskripsi Model:**
RNN adalah arsitektur jaringan saraf yang dirancang untuk menangani data berurutan seperti deret waktu. Model ini mampu mempertahankan informasi dari data sebelumnya melalui koneksi rekursif. Dalam proyek ini, Long Short-Term Memory (LSTM), varian dari RNN, digunakan untuk menangani masalah vanishing gradient.

3. **Recurrent Neural Network (RNN)**:
   **Parameter dan Arsitektur:**

- **Layer LSTM pertama:**
  - `units=50`: Jumlah neuron di dalam layer.
  - `activation='relu'`: Fungsi aktivasi untuk menangkap hubungan non-linear dalam data.
  - `return_sequences=True`: Mengizinkan pengembalian urutan lengkap, yang diperlukan untuk layer LSTM berikutnya.
- **Dropout:**
  - `rate=0.2`: Mengurangi overfitting dengan menonaktifkan 20% neuron selama pelatihan.
- **Layer LSTM kedua:**
  - `units=50` dan `activation='relu'`: Sama seperti layer pertama.
- **Output Dense layer:**
  - `units=1`: Menghasilkan satu output untuk prediksi target.

**Optimasi:**

- **Optimizer:** `adam`, algoritma berbasis gradient descent yang menggabungkan momentum dan adaptasi learning rate.
- **Fungsi loss:** `mse` (Mean Squared Error), cocok untuk masalah regresi.
- **Epochs:** 50 iterasi pelatihan.
- **Batch size:** 32, menentukan jumlah sampel dalam setiap update parameter.

## **Peningkatan Model**

1. **Hyperparameter Tuning:**

   - Hyperparameter tuning dilakukan menggunakan `GridSearchCV` untuk model Random Forest dan Gradient Boosting guna mencari kombinasi parameter optimal seperti jumlah pohon (`n_estimators`), depth maksimum pohon (`max_depth`), dan learning rate (untuk Gradient Boosting).

2. **Regularisasi pada RNN:**

   - Dropout layer digunakan untuk mengurangi overfitting dengan mengabaikan sebagian neuron selama proses pelatihan.

3. **Validasi:**
   - Data validasi digunakan untuk memonitor kinerja model selama pelatihan dan mencegah overfitting pada data pelatihan.

#### Peningkatan Model:

- Hyperparameter tuning menggunakan GridSearchCV.
- Menambahkan lapisan Dropout pada RNN untuk mengurangi overfitting.

---

### Evaluation

#### Metrik Evaluasi:

- **Mean Absolute Error (MAE)**: Mengukur rata-rata kesalahan absolut antara nilai aktual dan prediksi.
- **Mean Squared Error (MSE)**: Memberikan penalti lebih besar untuk kesalahan yang besar.
- **R-squared (R²)**: Mengukur seberapa baik model menjelaskan variasi data target.

#### Hasil Evaluasi:

| Model                           | MAE    | MSE     | R²     |
| ------------------------------- | ------ | ------- | ------ |
| **Random Forest Regressor**     | 0.0106 | 0.0002  | 0.9946 |
| **Gradient Boosting Regressor** | 0.0102 | 0.00018 | 0.9951 |
| **RNN**                         | 0.0368 | 0.0023  | 0.9381 |

#### Insight:

- **RNN** menunjukkan performa terbaik dengan nilai **R²** 0.9381, karena kemampuannya dalam menangkap pola deret waktu.
- **Gradient Boosting Regressor** menawarkan keseimbangan antara akurasi dan waktu pelatihan, dengan **R²** yang sedikit lebih tinggi dari **Random Forest**.
- **Random Forest** memberikan hasil yang sangat baik dengan **R²** 0.9946, tetapi lebih cepat dibandingkan RNN.

---

### Rekomendasi

1. **Peningkatan Data**: Mengumpulkan lebih banyak data historis untuk meningkatkan akurasi model.
2. **Eksperimen dengan Arsitektur LSTM**: Menerapkan Long Short-Term Memory (LSTM) untuk menangkap pola deret waktu yang lebih kompleks.
3. **Integrasi Model**: Mengintegrasikan model prediksi cuaca ke dalam aplikasi berbasis web atau mobile untuk memberikan nilai tambah bagi pengguna.

---
