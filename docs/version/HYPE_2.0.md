# HyperGrid Model v2.0 (Liquid Neural Network - CfC)

Dokumentasi teknis untuk arsitektur continuous-time deep learning generasi ketiga HyperGrid berbasis Closed-form Continuous-Time Networks (CfC).

## Ringkasan Model

- Tipe Model: Liquid Neural Network (CfCLiquidNet) berbasis formulasi analitik tertutup ODE (Hasani et al., Nature Machine Intelligence 2022)
- Input: Sekuens telemetri 8-channel mentah `(batch, sequence_length=100, channels=8)` beserta delta waktu inter-sample `dt` `(batch, sequence_length=100, 1)`
- Output: Klasifikasi multi-kelas 4 kategori kondisi jalan:
  - Kelas 0: Normal Asphalt / Smooth Road
  - Kelas 1: Pothole / Severe Drop Impact
  - Kelas 2: Speed Bump / Asphalt Ridge
  - Kelas 3: Rough / Cobblestone / Uneven Pavement
- Runtime Target: PyTorch Checkpoint (`models/cfc/best_cfc_model.pt`), ONNX Runtime (`models/cfc/hype_liquid_v3.onnx`), dan TensorFlow Lite Float32 / Float16 / INT8 (`models/cfc/hype_liquid_v3.tflite`)

## Arsitektur Jaringan (CfCLiquidNet)

Model dibangun dari tiga blok pemrosesan utama:

1. **Frontend 1D Depthwise-Separable Convolution**:
   - Menerima tensor input 8 kanal (6-DoF akselerometer/giroskop + 2 norma magnitudo L2).
   - Mengekstrak pola getaran frekuensi tinggi (*micro-vibration shockwaves*) sebelum masuk ke pemrosesan temporal dengan kompleksitas komputasi yang jauh lebih rendah dibanding konvolusi standar.
2. **Analytic CfC Recurrent Layer**:
   - Menghitung evolusi continuous-time ODE secara analitik tanpa numerik solver (Runge-Kutta atau Euler).
   - Menerima interval waktu nyata $\Delta t$ per sampel untuk menangani sampling jitter pada sensor smartphone.
   - Menggunakan 32 unit recurrent state dengan waktu inferensi konstan O(1) per langkah waktu.
3. **Temporal Attention Pooling**:
   - Menggantikan pooling rata-rata biasa dengan mekanisme atensi dinamis yang memusatkan pembobotan pada lonjakan impuls benturan transien (lubang atau gundukan).
   - Dilanjutkan dengan Linear Classifier head dan dropout regularisasi.

## Pipeline Data dan Preprocessing

- **Penyelarasan Delta Waktu ($\Delta t$)**: Setiap frame telemetri mencatat selisih timestamp aktual ke milidetik presisi untuk memandu dinamika pembusukan membran neuron kontinu.
- **Normalisasi Kuat (Robust Sensor Scaler)**: Menggunakan median centering dan Interquartile Range (IQR) per kanal sensor untuk menghilangkan bias kalibrasi antar perangkat smartphone.
- **Augmentasi Sinyal Temporal**: Pelatihan memanfaatkan kombinasi augmentasi jitter Gaussian, penskalaan suspensi kendaraan, distorsi interpolasi waktu, dan rotasi spasial 3D terhadap vektor gravitasi.

## Strategi Pelatihan dan Hasil Evaluasi

- **Loss Function**: Multi-Class Focal Loss dengan pembobotan frekuensi terbalik untuk mengatasi ketimpangan jumlah sampel antara jalan normal dan lubang.
- **Validasi**: 5-Fold Stratified GroupKFold cross-validation berbasis ID perjalanan unik (`trip_id`).
- **Kinerja Validasi**:
  - Macro F1-Score: 0.7978 (terbaik), rata-rata 0.7563 di seluruh fold
  - Rata-rata Precision: 0.7591
  - Rata-rata Recall: 0.7867
  - Normal Road False Positive Rate (FPR): 15.26% (tingkat alarm palsu pada jalan mulus)

## Profil Ekspor dan Edge Deployment

- **Ukuran Model**: Bobot model berukuran ringkas (< 350 KB pada format 32-bit floating point, < 100 KB pada kuantisasi INT8).
- **Kompatibilitas Runtime**: Tanpa dependensi operator kustom (*zero custom ops*), sepenuhnya kompatibel dengan TensorFlow Lite FlatBuffers dan ONNX Runtime.
- **Latensi Edge**: Inferensi single-window berkisar di bawah 1.0 milidetik pada CPU perangkat seluler.
