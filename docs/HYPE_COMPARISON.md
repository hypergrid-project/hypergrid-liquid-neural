# Perbandingan Arsitektur Model HyperGrid: v1.0, v1.2, dan v2.0

Dokumen ini membandingkan evolusi teknis arsitektur model deteksi kondisi jalan HyperGrid lintas tiga generasi utama.

## Ringkasan Matriks Perbandingan

| Dimensi Arsitektur | HyperGrid v1.0 | HyperGrid v1.2 | HyperGrid v2.0 (LNN) |
| :--- | :--- | :--- | :--- |
| **Paradigma Model** | Tabular ML (XGBoost) | Tabular ML (XGBoost) + MLP Proxy | Continuous-Time Deep Learning (CfCLiquidNet) |
| **Representasi Input** | Fitur statistik agregat (mean, std, rms, zcr) | 30 fitur dekomposisi getaran vertikal/horizontal | Sekuens telemetri mentah 8 kanal + interval $\Delta t$ per langkah |
| **Ukuran Window** | 20 sampel (~4.0 detik pada 5 Hz) | 15 sampel (~3.0 detik pada 5 Hz) | 100 sampel (~2.0 detik pada 50 Hz nominal) |
| **Target Prediksi** | Biner (Normal, Pothole) | Biner (Normal, Pothole) + Confidence Tier | 4 Kelas (Smooth, Pothole, Speed Bump, Rough Road) |
| **Penanganan Sampling Jitter** | Diasumsikan sampling seragam konstan | Diasumsikan sampling seragam konstan | Eksplisit dimodelkan via variabel continuous-time $\Delta t$ |
| **Estimasi Gravitasi** | Rolling median (non-kausal) | Exponential Moving Average (kausal, alpha=0.1) | Invarian rotasi 3D dan Magnitudo L2 norm |
| **Pooling Temporal** | Agregasi statistik manual di awal | Agregasi statistik manual di awal | Learned Temporal Attention Pooling |
| **Fungsi Objektif / Loss** | Binary Logistic Loss | Binary Logistic Loss berbobot | Multi-Class Focal Loss ($\gamma = 2.0$) |
| **Format Artefak Edge** | ONNX Runtime | ONNX Runtime dan TFLite (via MLP proxy) | PyTorch, ONNX, TFLite (Float32, Float16, INT8) |

## Analisis Komparatif Per Komponen

### 1. Representasi Sinyal dan Rekayasa Fitur

- **HyperGrid v1.0**: Menghitung fitur statistik dasar pada magnitudo total akselerasi dan rotasi giroskop. Kelemahan pendekatan ini adalah hilangnya arah vektor gaya serta dinamika transien cepat.
- **HyperGrid v1.2**: Memperkenalkan dekomposisi vektor proyeksi gravitasi menjadi komponen akselerasi vertikal ($a_v$) dan planar horizontal ($a_h$). Pendekatan ini secara signifikan mengurangi alarm palsu dari manuver belokan atau pengereman kendaraan.
- **HyperGrid v2.0**: Menghilangkan tahap *feature engineering* manual. Model menerima langsung deret waktu 8 kanal mentah ($A_x, A_y, A_z, G_x, G_y, G_z, \|A\|_2, \|G\|_2$). Frontend 1D Depthwise-Separable Convolution mengekstrak pola gelombang kejut mikroskopis secara end-to-end.

### 2. Penanganan Ketidakteraturan Waktu (Time-Jitter Resilience)

- **v1.0 dan v1.2**: Bergantung pada asumsi frekuensi sampling konstan. Fluktuasi interval waktu akibat *thread scheduling* OS smartphone atau *clock drift* dapat mendistorsi hasil estimasi turunan sinyal (seperti *jerk*).
- **v2.0**: Memanfaatkan formulasi *Closed-form Continuous-Time* (CfC). Peluruhan potensial membran dan pembaruan *state* dihitung langsung terhadap selisih waktu aktual $\Delta t_t$, sehingga model stabil terhadap variasi frekuensi sampling sensor di lapangan.

### 3. Kausalitas dan Kesiapan Edge Deployment

- **v1.0**: Penggunaan *rolling median* dua arah (non-kausal) menyebabkan disparitas kecil (*train-serve skew*) antara kalkulasi batch saat pelatihan dan inferensi titik streaming.
- **v1.2**: Memperbaiki kausalitas menggunakan filter Exponential Moving Average (EMA). Untuk eksekusi pada perangkat seluler berdaya rendah, disediakan modul proxy Keras MLP yang diekspor ke TensorFlow Lite.
- **v2.0**: Didesain dari awal tanpa ketergantungan *custom operators* (*zero custom ops*). Formulasi CfC analitik tidak memerlukan integrasi numerik solver Runge-Kutta, sehingga graph dapat diekspor langsung ke format TFLite dan ONNX flat ops dengan latensi di bawah 1.0 ms.

### 4. Lingkup Klasifikasi Permukaan Jalan

- **v1.0 dan v1.2**: Terbatas pada pendeteksian biner apakah segmen jalan memiliki lubang (*pothole*) atau tidak. Kategori anomali lain seperti polisi tidur (*speed bump*) berpotensi terdeteksi sebagai lubang akibat lonjakan akselerasi.
- **v2.0**: Memperluas pemetaan ke 4 kategori kondisi jalan secara simultan: aspal mulus (Normal), benturan lubang (*pothole*), punggung jalan (*speed bump*), dan jalan bergelombang atau berbatu (*rough road*).

## Rekomendasi Penggunaan

- **Pilih v1.2** jika sistem target memiliki keterbatasan lingkungan eksekusi Python berbasis CPU server sederhana dan membutuhkan interpretabilitas berbasis aturan pohon keputusan (XGBoost).
- **Pilih v2.0 (LNN CfC)** untuk implementasi produksi modern, integrasi aplikasi mobile / on-device edge computing langsung pada smartphone pengemudi, atau sistem yang membutuhkan klasifikasi granular multi-kelas dengan toleransi tinggi terhadap ketidakteraturan interval telemetri sensor.
