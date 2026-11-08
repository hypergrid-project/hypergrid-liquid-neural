# HyperGrid Model v1.2

Dokumentasi teknis untuk model deteksi anomali permukaan jalan HyperGrid generasi kedua.

## Ringkasan Model

- Tipe Model: Gradient Boosted Trees (XGBoost Classifier) dan Proxy MLP untuk Edge TFLite
- Input: 30 fitur turunan berbasis dekomposisi getaran vertikal dan horizontal kausal
- Output: Klasifikasi biner (0: Normal Road, 1: Pothole) disertai Confidence Tiers (HIGH, MEDIUM, LOW)
- Runtime Target: ONNX Runtime (`models/hype_v2.onnx`) dan TensorFlow Lite (`models/hype_v2.tflite`)

## Pembaruan dari Versi 1.0

Versi 1.2 menyelesaikan masalah ketidaksesuaian runtime (*train-serve skew*) dan menyempurnakan kualitas sinyal:

1. **Pembersihan Telemetri Lanjutan**:
   - Deteksi dan koreksi kecepatan GPS negatif akibat keterlambatan kalkulasi sensor satelit.
   - Pengecekan monotonisitas timestamp untuk mencegah regresi sekuensial.
2. **Estimasi Gravitasi Kausal (Causal EMA)**:
   - Menggantikan rolling median non-kausal dengan Exponential Moving Average (koefisien alpha = 0.1).
   - Menghasilkan vektor arah gravitasi dinamis yang identik baik pada training offline maupun inferensi live streaming.
3. **Dekomposisi Sinyal Vertikal dan Horizontal**:
   - Memproyeksikan sinyal akselerasi ke vektor gravitasi untuk mengisolasi getaran vertikal ($a_v$).
   - Menghitung komponen planar horizontal ($a_h$) untuk membedakan tikungan tajam atau pengereman dari lubang jalan.
4. **Normalisasi Intensitas Kecepatan**:
   - Menyesuaikan amplitudo guncangan dinamis terhadap rasio kecepatan saat kendaraan melaju.

## Parameter Windowing & Fitur

- Window size: 15 sampel (~3.0 detik pada frekuensi 5 Hz)
- Step size / Stride: 3 sampel (~0.6 detik per geser)
- Fitur yang diekstrak:
  - Statistik gelombang pada sinyal vertikal $a_v$ dan horizontal $a_h$.
  - Turunan getaran sekunder (jerk vertikal, jerk horizontal).
  - Rasio rasio energi guncangan terhadap kecepatan kendaraan.

## Skema Confidence Tiers

Model menghitung probabilitas prediksi dan mengelompokkan hasil inferensi ke dalam tiga tingkat keyakinan:
- **HIGH** (Probabilitas >= 0.80): Indikasi lubang dengan keyakinan tinggi.
- **MEDIUM** (Probabilitas 0.50 s.d. 0.79): Indikasi anomali sedang, memerlukan konfirmasi dari trip berulang.
- **LOW** (Probabilitas < 0.50): Permukaan normal atau gelombang jalan biasa.

## Model Edge TFLite

Untuk mendukung eksekusi ringan di perangkat seluler tanpa modul library XGBoost C++, versi ini menyediakan proxy Multi-Layer Perceptron (Keras MLP) dengan layer normalisasi internal yang diekspor langsung ke format TensorFlow Lite.
