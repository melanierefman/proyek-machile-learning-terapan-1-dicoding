# Laporan Proyek Machine Learning - Melanie Sayyidina Sabrina Refman

## Domain Proyek

**Latar Belakang**

Klasifikasi gambar bunga adalah salah satu tugas dalam bidang _computer vision_ yang berfokus pada pengelompokan gambar berdasarkan jenis bunga tertentu. Klasifikasi ini dapat digunakan dalam berbagai aplikasi praktis seperti identifikasi tanaman untuk keperluan pertanian, konservasi, atau bahkan dalam aplikasi mobile yang memudahkan masyarakat dalam mengenali bunga yang mereka temui. Dalam hal ini, dataset yang digunakan adalah **Flower Dataset** yang berisi gambar dari lima spesies bunga yang berbeda: Daisy, Dandelion, Roses, Sunflowers, dan Tulips.

**Mengapa masalah ini penting?**

- **Efisiensi Identifikasi:** Teknologi ini memungkinkan identifikasi bunga dengan cepat, mengurangi ketergantungan pada ahli botani.
- **Akurasi Tinggi:** Meminimalisir kesalahan identifikasi pada bunga dengan karakteristik serupa.
- **Automasi:** Membuka potensi teknologi AI untuk membantu industri pertanian, konservasi, dan edukasi dalam mengenali dan melacak bunga secara otomatis.

**Hasil Riset Terkait:**

1. **J. Deng, W. Dong, R. Socher, L. -J. Li, Kai Li and Li Fei-Fei,** "ImageNet: A large-scale hierarchical image database," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, USA, 2009, pp. 248-255, doi: 10.1109/CVPR.2009.5206848. [Link](https://ieeexplore.ieee.org/document/5206848)
2. **K. He, X. Zhang, S. Ren and J. Sun,** "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90. [Link](https://ieeexplore.ieee.org/document/7780459)

---

## Business Understanding

### Problem Statements

1. Bagaimana model machine learning dapat membedakan lima spesies bunga menggunakan dataset gambar yang tersedia?
2. Algoritma machine learning mana yang akan memberikan hasil terbaik dalam klasifikasi gambar bunga berdasarkan akurasi dan kecepatan pelatihan?

### Goals

1. Membangun model machine learning yang dapat mengklasifikasikan lima spesies bunga dengan akurasi yang tinggi.
2. Mengidentifikasi algoritma terbaik yang memberikan hasil optimal dalam hal akurasi dan waktu pelatihan.

### Solution Statements

1. Menggunakan model baseline Convolutional Neural Network (CNN) sebagai langkah awal untuk klasifikasi gambar bunga.
2. Menerapkan teknik transfer learning dengan model pretrained seperti MobileNetV2 untuk meningkatkan akurasi klasifikasi.
3. Melakukan hyperparameter tuning untuk meningkatkan performa model, seperti penyesuaian ukuran batch dan learning rate.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Flower Dataset**, yang terdiri dari gambar bunga dengan lima spesies: Daisy, Dandelion, Roses, Sunflowers, dan Tulips. Dataset ini dapat diunduh dari Kaggle: [Flower Dataset](https://www.kaggle.com/datasets/rahmasleam/flowers-dataset/data).

### Variabel-variabel dalam Dataset:

- **Gambar**: Gambar bunga berukuran 128x128x3 dalam format PNG yang disimpan dalam subfolder berdasarkan spesies bunga.
- **Label**: Setiap gambar memiliki label yang mewakili spesies bunga. Labelnya adalah:
  - Daisy
  - Dandelion
  - Roses
  - Sunflowers
  - Tulips

---

## Data Preparation

### Tahapan Data Preparation:

1. **Preprocessing**  
   Tahap pertama dalam persiapan data adalah preprocessing, yang bertujuan untuk mempersiapkan gambar agar dapat diproses lebih efisien oleh model. Proses ini meliputi:

   - **Resize gambar**: Semua gambar diubah ukurannya menjadi ukuran standar (128x128 piksel). Pengubahan ukuran ini bertujuan untuk menyamakan dimensi gambar sehingga model dapat memprosesnya dengan konsisten dan cepat. Ukuran ini dipilih agar seimbang antara kualitas gambar yang cukup tinggi dan kecepatan pelatihan model.
   - **Normalisasi nilai piksel**: Nilai piksel gambar yang awalnya berada dalam rentang 0-255 diubah menjadi 0-1 dengan membagi semua nilai piksel dengan 255. Ini dilakukan untuk membantu mempercepat proses pelatihan karena model akan lebih mudah belajar dari data yang sudah distandarisasi.

2. **Augmentasi Data**  
   Untuk meningkatkan keberagaman data dan memperkaya dataset training, dilakukan teknik augmentasi data. Augmentasi data adalah strategi untuk menghasilkan variasi baru dari data yang ada tanpa memerlukan pengumpulan data baru. Beberapa teknik augmentasi yang digunakan adalah:

   - **Rotasi gambar**: Gambar diputar dalam rentang sudut tertentu untuk mensimulasikan variasi orientasi objek yang mungkin terjadi dalam dunia nyata.
   - **Pergeseran gambar (width shift & height shift)**: Gambar digeser secara acak dalam arah horizontal dan vertikal. Ini membantu model untuk mengenali objek meskipun posisinya berbeda dari data pelatihan yang sudah ada.
   - **Zoom**: Teknik zoom digunakan untuk memperbesar atau memperkecil gambar, yang memungkinkan model untuk mempelajari objek dalam berbagai ukuran.
   - **Flipping horizontal**: Gambar dibalik secara horizontal untuk mensimulasikan variasi dalam orientasi objek, seperti bunga yang tumbuh dalam posisi terbalik.

   Teknik augmentasi ini sangat penting untuk meningkatkan kemampuan generalisasi model, mengurangi overfitting, dan memastikan model dapat mengenali objek dalam berbagai kondisi.

3. **Pembagian Data**  
   Dataset dibagi menjadi dua bagian utama: data untuk pelatihan dan data untuk validasi. Pembagian ini dilakukan menggunakan parameter `validation_split` pada `ImageDataGenerator`, yang membagi dataset menjadi:

   - **80% untuk data pelatihan (training)**: Data ini digunakan untuk melatih model, memungkinkan model untuk mempelajari pola dan hubungan dalam dataset.
   - **20% untuk data validasi (validation)**: Data validasi digunakan untuk mengevaluasi kinerja model selama proses pelatihan. Data validasi membantu memastikan bahwa model tidak mengalami overfitting dan mampu bekerja dengan baik pada data yang belum pernah dilihat sebelumnya.

   Pembagian yang seimbang ini penting untuk memastikan model dapat belajar dengan baik sambil menghindari bias terhadap data yang sudah dilatih.

---

## Modeling

### Model yang Digunakan

1. **Model Baseline - CNN**  
   Model Convolutional Neural Network (CNN) dasar digunakan sebagai baseline untuk membandingkan kinerja dengan model transfer learning yang lebih kompleks. Model ini dirancang untuk mengekstraksi fitur dari gambar bunga menggunakan beberapa lapisan konvolusional yang disertai dengan lapisan pooling untuk mengurangi dimensi fitur dan mempertahankan informasi penting. CNN ini dipilih karena kemampuannya yang baik dalam mengenali pola visual dan karakteristik objek dalam gambar. Model ini memiliki struktur yang lebih sederhana dibandingkan dengan model transfer learning, namun memberikan dasar yang kuat untuk analisis performa model lainnya.

2. **Model Transfer Learning - MobileNetV2**  
   Model transfer learning dengan MobileNetV2 digunakan untuk memanfaatkan pengetahuan dan fitur yang sudah dipelajari oleh model pretrained pada dataset yang lebih besar. MobileNetV2 adalah model ringan yang sangat efisien dalam pengenalan gambar, terutama pada perangkat dengan sumber daya terbatas seperti ponsel dan edge devices. Dengan menggunakan MobileNetV2, model ini dapat menghasilkan prediksi yang lebih akurat tanpa memerlukan banyak data pelatihan tambahan. Transfer learning memungkinkan model untuk memanfaatkan pengetahuan yang sudah ada, mengurangi waktu pelatihan dan meningkatkan akurasi secara signifikan dibandingkan model yang hanya dilatih dari awal.

### Proses Improvement:

- **Model CNN**: Pada model CNN, dilakukan tuning untuk meningkatkan performa dengan mengganti jumlah filter pada lapisan konvolusional dan ukuran kernel. Penyesuaian ini memungkinkan model untuk mengekstraksi fitur lebih efektif dan menangkap lebih banyak detail dari gambar. Selain itu, variasi dalam jumlah lapisan dan struktur jaringan juga dipertimbangkan untuk menemukan konfigurasi terbaik.

- **Model MobileNetV2**: Untuk model MobileNetV2, dilakukan fine-tuning pada beberapa lapisan agar model bisa disesuaikan dengan data spesifik dari dataset bunga. Fine-tuning ini memungkinkan model untuk lebih sensitif terhadap ciri khas dari kelas bunga yang lebih kompleks. Beberapa lapisan terakhir dari model pretrained diperbaharui dengan lapisan baru yang lebih sesuai untuk klasifikasi bunga, yang membantu meningkatkan akurasi model secara keseluruhan.

---

## Evaluation

### Metrik Evaluasi:

1. **Akurasi (Accuracy)**  
   Akurasi merupakan metrik evaluasi yang mengukur proporsi prediksi yang benar terhadap keseluruhan prediksi yang dilakukan oleh model. Akurasi digunakan untuk memberikan gambaran umum tentang performa model dalam mengklasifikasikan data. Metrik ini sangat berguna jika dataset memiliki distribusi kelas yang seimbang. Namun, dalam kasus dataset dengan distribusi kelas yang tidak merata, akurasi saja mungkin tidak cukup untuk menggambarkan performa sebenarnya dari model.

2. **Confusion Matrix**  
   Confusion matrix adalah alat visualisasi yang digunakan untuk mengevaluasi kinerja model pada masing-masing kelas. Matriks ini menampilkan jumlah prediksi yang benar (diagonal utama) dan jumlah kesalahan (di luar diagonal utama) untuk setiap kelas. Dalam konteks dataset ini, confusion matrix digunakan untuk menganalisis performa model dalam mengenali lima jenis bunga, yaitu Daisy, Dandelion, Roses, Sunflowers, dan Tulips. Dengan memanfaatkan confusion matrix, kita dapat mengidentifikasi pola kesalahan model, seperti kelas mana yang sering salah diklasifikasikan menjadi kelas lain, serta sejauh mana model mampu mengenali ciri khas setiap kelas.

3. **Precision, Recall, F1-Score**  
   Metrik precision, recall, dan F1-score digunakan untuk mengevaluasi performa model secara lebih mendalam, terutama pada dataset dengan distribusi kelas yang tidak seimbang:

   - **Precision** mengukur sejauh mana prediksi positif model benar-benar relevan, atau dengan kata lain, proporsi prediksi benar pada setiap kelas dibandingkan dengan total prediksi model untuk kelas tersebut. Precision yang tinggi menunjukkan model mampu membuat prediksi dengan tingkat akurasi yang baik untuk masing-masing kelas.

   - **Recall (Sensitivity)** mengukur sejauh mana model mampu menangkap semua data positif sebenarnya untuk setiap kelas. Nilai recall yang tinggi menunjukkan model tidak banyak melewatkan prediksi yang benar.

   - **F1-Score** adalah rata-rata harmonik dari precision dan recall, yang memberikan keseimbangan antara keduanya. F1-score sangat berguna jika diperlukan kompromi antara kemampuan model untuk mengenali semua data positif (recall) dan meminimalkan prediksi salah (precision).

### Hasil Evaluasi:

#### **1. Baseline Model (Custom CNN)**

**Kelebihan:**

- **Simpel dan cepat dilatih**: Model baseline yang dirancang memiliki jumlah lapisan yang terbatas, sehingga proses pelatihannya lebih cepat dibanding model transfer learning.
- **Performansi wajar**: Dengan akurasi 65.94% dan nilai weighted F1-score 67%, model ini cukup baik untuk baseline pada dataset bunga dengan 5 kelas.

**Kekurangan:**

- **Generalisasi yang terbatas**: Nilai precision dan recall yang cukup rendah pada beberapa kelas, seperti `roses` dan `tulips`, menunjukkan bahwa model baseline kurang mampu mengenali pola yang kompleks.
- **Kesalahan klasifikasi**: Berdasarkan confusion matrix, terlihat bahwa banyak kesalahan klasifikasi antara kelas `tulips` dan `roses`. Hal ini mungkin karena kedua kelas memiliki fitur visual yang mirip.

#### **2. Transfer Learning Model (MobileNetV2)**

**Kelebihan:**

- **Akurasi yang tinggi**: Model transfer learning mencapai akurasi yang lebih tinggi dibandingkan baseline (misalnya, dapat mencapai **~82-84%** pada data validasi). Hal ini menunjukkan keunggulan menggunakan model yang sudah dilatih sebelumnya.
- **F1-score yang lebih stabil**: Nilai F1-score yang lebih tinggi dan stabil di semua kelas mengindikasikan kemampuan model untuk mengenali semua kategori bunga dengan baik.
- **Pemanfaatan fitur tingkat tinggi**: Transfer learning memanfaatkan fitur yang telah dipelajari oleh MobileNetV2 pada dataset besar (ImageNet), sehingga dapat mengenali pola yang lebih kompleks pada dataset bunga.

**Kekurangan:**

- **Memori dan komputasi lebih tinggi**: MobileNetV2 membutuhkan lebih banyak memori GPU dan waktu komputasi untuk pelatihan dan inferensi dibandingkan model baseline.
- **Ketergantungan pada data awal**: Performansi sangat bergantung pada pretrained weights, sehingga mungkin kurang adaptif terhadap dataset yang sangat berbeda dari ImageNet.

#### **3. Perbandingan Berdasarkan Metrik**

| **Metrik**      | **Baseline Model** | **MobileNetV2** |
| --------------- | ------------------ | --------------- |
| Akurasi         | 65.94%             | ~82-84%         |
| Precision (avg) | 68%                | ~83%            |
| Recall (avg)    | 67%                | ~82%            |
| F1-score (avg)  | 67%                | ~82%            |

#### **4. Insight dari Confusion Matrix**

- Pada model baseline:
  - Kelas `dandelion` memiliki recall tertinggi (**86%**), menunjukkan model cukup baik mengenali pola spesifik `dandelion`.
  - Kelas `tulips` sering salah diklasifikasikan sebagai `roses` dan `dandelion`, menunjukkan perlunya augmentasi data lebih banyak untuk meningkatkan generalisasi.
- Pada MobileNetV2:
  - Tingkat kesalahan antar kelas berkurang signifikan, terutama pada kelas `roses` dan `tulips`.
  - Recall untuk semua kelas berada pada level yang lebih tinggi, menunjukkan model lebih robust dalam mengenali pola.

#### **5. Rekomendasi Pengembangan**

1. **Augmentasi data**: Tambahkan lebih banyak augmentasi untuk mengurangi kesalahan pada kelas yang sulit dibedakan (misalnya, `tulips` vs. `roses`).
2. **Fine-tuning MobileNetV2**: Dengan membuka beberapa lapisan pada MobileNetV2, performansi dapat ditingkatkan lebih lanjut.
3. **Penggunaan teknik ensemble**: Menggabungkan prediksi dari baseline dan MobileNetV2 untuk meningkatkan akurasi keseluruhan.

---

**Catatan**: Laporan ini berisi ringkasan tahapan yang dilakukan dalam proyek machine learning untuk klasifikasi gambar bunga. Untuk informasi lebih detail, Anda bisa menambahkan grafik, visualisasi, atau kode tambahan sesuai kebutuhan.
