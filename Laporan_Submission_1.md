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

Dataset mencakup data cuaca harian dari 18 lokasi di Eropa antara tahun 2000 hingga 2010. Variabel utama dalam dataset meliputi:

- **Suhu**: Rata-rata, minimum, dan maksimum suhu harian dalam derajat Celsius.
- **Kelembapan**: Persentase kelembapan udara.
- **Kecepatan Angin**: Dalam meter per detik (m/s).
- **Tekanan Udara**: Dalam 1000 hPa.
- **Curah Hujan**: Dalam sentimeter (cm).
- **Durasi Sinar Matahari**: Dalam jam.

#### Distribusi Data

Data tersedia dalam bentuk deret waktu dengan total 3654 observasi harian. Beberapa variabel memiliki nilai yang hilang, namun telah diatasi dengan metode imputasi.

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

- **Random Forest Regressor**: Algoritma berbasis pohon keputusan yang kuat dalam menangani data non-linear dan interaksi antar fitur.
- **Gradient Boosting Regressor**: Model boosting yang membangun pohon secara iteratif untuk meminimalkan kesalahan prediksi.
- **Recurrent Neural Networks (RNN)**: Arsitektur jaringan saraf untuk data berurutan, cocok untuk deret waktu cuaca.

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
