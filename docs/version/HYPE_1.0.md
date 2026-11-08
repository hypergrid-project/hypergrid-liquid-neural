# HyperGrid Model v1.0

Dokumentasi teknis untuk model deteksi anomali permukaan jalan HyperGrid generasi pertama.

## Ringkasan Model

- Tipe Model: Gradient Boosted Trees (XGBoost Classifier)
- Input: Ekstraksi fitur statistik berbasis sliding window dari sensor IMU smartphone
- Output: Klasifikasi biner (0: Normal Road, 1: Pothole)
- Runtime Target: ONNX Runtime (`models/hype_v1.onnx`)

## Pipeline Pemrosesan Data

1. **Sinkronisasi Sinyal**: Penyelarasan timestamp antara sensor akselerometer 3 sumbu, giroskop 3 sumbu, dan koordinat GPS.
2. **Kompensasi Orientasi**: Perhitungan magnitudo akselerasi dan giroskop:
   - Akselerasi total: $\sqrt{A_x^2 + A_y^2 + A_z^2}$
   - Giroskop total: $\sqrt{G_x^2 + G_y^2 + G_z^2}$
3. **Penyaringan Gravitasi**: Kompensasi gravitasi statis menggunakan rolling median baseline untuk mengisolasi getaran dinamis kendaraan (`dynamic_acc`).
4. **Sliding Window**:
   - Window size: 20 sampel (~4.0 detik pada sampling rate 5 Hz)
   - Step size / Stride: 5 sampel (~1.0 detik per geser)

## Ekstraksi Fitur

Model mengekstrak metrik statistik agregat dari setiap jendela:
- Sinyal yang diproses: `dynamic_acc`, `gyro_mag`, `jerk`, `angular_jerk`, `speed`
- Metrik per sinyal: Mean, Standard Deviation, Minimum, Maximum, Range, Root Mean Square (RMS)
- Metrik impuls: Crest Factor dan Zero-Crossing Rate (ZCR)

## Pelatihan dan Validasi

- Skema Validasi: GroupKFold cross-validation berbasis ID perjalanan (`trip_id`) untuk mencegah kebocoran data antar segmen jalan yang sama.
- Penyetelan Hyperparameter: Optimasi parameter pohon keputusan XGBoost melalui evaluasi ROC-AUC pada data uji perjalanan unseen.
- Format Model: Konversi dari model XGBoost terlatih ke format ONNX menggunakan library `onnxmltools`.

## Keterbatasan Versi 1.0

- Perhitungan rolling median baseline non-kausal menimbulkan sedikit ketidaksesuaian antara mode evaluasi batch dan streaming waktu nyata.
- Nilai agregat statistik jendela menghilangkan karakteristik transien mikroskopis sinyal benturan.
- Belum memisahkan gaya akselerasi vertikal terhadap guncangan horizontal kendaraan.
