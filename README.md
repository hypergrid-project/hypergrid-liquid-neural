# Hypergrid Liquid Neural

Model *continuous-time deep learning* berbasis *Liquid Neural Networks* (CfC) dan *inertial machine learning* berbobot ringan untuk deteksi anomali permukaan jalan, lubang (*potholes*), polisi tidur (*speed bump*), dan jalan rusak berbasis data telemetri 6-DoF IMU smartphone secara *real-time*.

```bash
# Menjalankan pipeline pelatihan & pengujian model v2.0 (CfC Liquid)
uv run hype/autorun_hype_series_2.0.py

# Menjalankan benchmark dan evaluasi lintas versi
uv run hype/autorun_hype_series_1.2.py

```

---

## Model Tags & Variants

| Tag | Parameter / Ukuran | Arsitektur | Format Target | Karakteristik Utama | Status Edge |
| --- | --- | --- | --- | --- | --- |
| `latest`, `2.0`, `cfc` | ~45K param (204.1 KB) | Closed-form Continuous (CfC) | ONNX Runtime, PyTorch | 4-Class multi-event, toleran *clock jitter* ($\Delta t$), konvolusi depthwise-separable | **Production Ready** (ONNX) |
| `2.0-tflite` | 1.59 MB (FP32) / 0.98 MB (FP16) | CfC Graph via `onnx2tf` | TFLite FP32 / FP16 | Eksperimental; memerlukan resolusi custom op `ONNX_GEMM` | Review Pending |
| `1.2` | 70.6 KB | Causal Gravity EMA + MLP Proxy | ONNX Runtime | 3-Tier Confidence (HIGH, MED, LOW), dekomposisi getaran vertikal-horizontal | **Production Ready** |
| `1.0` | ~120 KB | Feature Extraction + XGBoost | ONNX Runtime | Baseline tabular biner 5 Hz, kompensasi gravitasi rolling median | Legacy Baseline |

---

## Quickstart

### 1. Inisialisasi Dependensi

Proyek dikelola menggunakan paket manajer `uv` untuk determinisme lingkungan runtime:

```bash
# Setup virtual environment dan sinkronisasi pustaka
uv sync

# Opsi alternatif jika menggunakan pip standar
pip install -r requirements.txt

```

### 2. Inferensi Cepat (Python API via PyTorch)

Memuat bobot dan menjalankan klasifikasi pada jendela sinyal 6-DoF mentah berdimensi `(Batch, Sequence, Features)`:

```python
import torch
from utils.model_cfc import CfCLiquidNet

# Inisialisasi model 8 kanal (6-DoF + a_v magnitude + delta_t)
model = CfCLiquidNet(in_features=8, num_classes=4)
model.load_state_dict(torch.load("models/hype_2.0.pt", map_location="cpu"))
model.eval()

# Input dummy: batch=1, window=100 frame (~1.0–2.0s), channels=8
# Format kanal: [Ax, Ay, Az, Gx, Gy, Gz, a_v, dt]
sample_stream = torch.randn(1, 100, 8)

with torch.no_grad():
    logits = model(sample_stream)
    prediction = torch.argmax(logits, dim=-1).item()

labels = {
    0: "Normal Asphalt",
    1: "Pothole Drop",
    2: "Speed Bump",
    3: "Rough Surface"
}

print(f"Kondisi Permukaan: {labels[prediction]}")

```

### 3. Inferensi Cepat (ONNX Runtime Mobile Engine)

```python
import onnxruntime as ort
import numpy as np

# Inisialisasi runtime engine berlatensi ultra-rendah
session = ort.InferenceSession("models/hype_2.0.onnx")
input_name = session.get_inputs()[0].name

# Dummy stream 8-kanal: (1, 100, 8) tipe float32
input_data = np.random.randn(1, 100, 8).astype(np.float32)
outputs = session.run(None, {input_name: input_data})

predicted_idx = int(np.argmax(outputs[0], axis=-1)[0])
print(f"Prediksi Kelas (ONNX): {predicted_idx}")

```

---

## Arsitektur & Cara Kerja (HyperGrid v2.0)

HyperGrid v2.0 menggantikan tahapan ekstraksi statistik manual dengan arsitektur *continuous-time recurrent* berbasis solusi diferensial analitik tertutup (**CfC - Closed-form Continuous-depth**):

```text
[Raw 6-DoF + Δt Stream] (8 Channels, Variable Freq)
           │
           ▼
[1D Depthwise-Separable Conv] ── Ekstraksi korelasi temporal lokal antar-kanal tanpa beban parameter
           │
           ▼
[CfC Recurrent Core]         ── Integrasi status membran kontinu terhadap interval waktu nyata (Δt)
           │
           ▼
[Temporal Attention Pooling] ── Pembobotan dinamis pada titik impak transien (lubang/polisi tidur)
           │
           ▼
[Linear Classifier]          ── Distribusi Probabilitas 4 Kelas (Softmax)

```

### Spesifikasi Kanal Input

Model mengonsumsi deret waktu dengan 8 saluran:

1. **$A_x$**: Akselerasi lateral perangkat (m/s²).
2. **$A_y$**: Akselerasi longitudinal perangkat (m/s²).
3. **$A_z$**: Akselerasi vertikal perangkat (m/s²).
4. **$G_x$**: Kecepatan sudut pitch (rad/s).
5. **$G_y$**: Kecepatan sudut roll (rad/s).
6. **$G_z$**: Kecepatan sudut yaw (rad/s).
7. **$a_v$**: Komponen akselerasi vertikal murni bebas gravitasi (estimasi filter EMA).
8. **$\Delta t$**: Selisih interval waktu aktual antar-sampel ($0.0010\text{ s} \le \Delta t \le 2.0000\text{ s}$) untuk mengoreksi *clock jitter*.

---

## Benchmark Empiris Lintas Generasi

Hasil pengujian diekstraksi dari log eksekusi validasi data riil (*unseen trips evaluation* pada 20 sesi perjalanan independen):

### 1. Karakteristik Arsitektur & Pelatihan

| Parameter | HyperGrid v1.0 | HyperGrid v1.2 | HyperGrid v2.0 (CfC Liquid) |
| --- | --- | --- | --- |
| **Paradigma** | Tabular Gradient Boosting | Tabular Gradient Boosting | Continuous-Time Deep Learning |
| **Representasi Input** | Fitur statistik manual (32 dim) | Dekomposisi vertikal/horizontal (47 dim) | Sinyal mentah 8 kanal + $\Delta t$ |
| **Dimensi Matriks** | $(4172, 32)$ | $(6975, 47)$ | $(770, 100, 8)$ |
| **Target Output** | Biner (Normal, Pothole) | Biner + 3-Tier Confidence | 4-Kelas (Normal, Pothole, Bump, Rough) |
| **Validasi Skema** | GroupKFold (5 Folds) | Stratified GroupKFold (5 Folds) | Stratified GroupKFold (5 Folds) |
| **Device Pelatihan** | CPU | CPU | Apple Silicon MPS (*Metal Performance Shaders*) |

### 2. Metrik Deteksi & Klasifikasi

| Metrik Evaluasi | HyperGrid v1.0 | HyperGrid v1.2 | HyperGrid v2.0 (CfC Liquid) |
| --- | --- | --- | --- |
| **ROC-AUC (Test/CV)** | 0.8724 | **0.8829** | *Multiclass Evaluation* |
| **PR-AUC (Avg Precision)** | *N/A* | 0.3815 | *N/A* |
| **Macro F1-Score** | 0.73 (Test) | 0.73 (Test) | **0.8172 (Mean CV)** / **0.9101 (Best Fold)** |
| **Pothole Precision** | 0.45 | 0.41 | **0.8254 (Mean Macro)** |
| **Pothole Recall** | 0.67 (Optimal Threshold) | **0.73** (Optimal Threshold) | **0.8525 (Mean Macro)** |
| **Optimal Threshold** | 0.473 | 0.316 | Softmax Argmax (0.50 standar) |
| **False Positive Rate** | 9.61% (110 / 1145) | 9.82% (192 / 1955) | 18.00% (Mean) / **4.00% (Best Epoch)** |
| **True Positive (Pothole)** | 89 titik | **131 titik** | Deteksi multikelas seimbang |

### 3. Efisiensi Runtime & Ukuran Model (Edge Deployment)

Pengukuran latensi CPU menggunakan pengujian berulang 1000 siklus:

| Metrik Deployment | HyperGrid v1.0 | HyperGrid v1.2 | HyperGrid v2.0 (CfC Liquid) |
| --- | --- | --- | --- |
| **Ukuran File ONNX** | ~120 KB | **70.6 KB** | 204.1 KB (Opset 16: 235.9 KB) |
| **Ukuran File TFLite FP32** | *Tidak tersedia* | *Tidak tersedia* | 1,625.86 KB (1.59 MB) |
| **Ukuran File TFLite FP16** | *Tidak tersedia* | *Tidak tersedia* | 1,004.95 KB (0.98 MB) |
| **Latensi Rata-rata (ONNX)** | ~1.20 ms | ~0.80 ms | **0.5756 ms** |
| **Latensi P95 / P99 (ONNX)** | *N/A* | *N/A* | **0.6131 ms / 0.8609 ms** |
| **Paritas Numerik (MAE)** | Identik | Identik | **1.689e-07 (PyTorch vs ONNX)** |
| **Status Runtime TFLite** | *N/A* | *N/A* | Unresolved Custom Op: `ONNX_GEMM` |

---

## Analisis Komparatif & Rekomendasi Deployment

1. **HyperGrid v2.0 (Rekomendasi Utama via ONNX Runtime)**:
* **Performa**: Menghasilkan lompatan akurasi signifikan dengan **Mean Macro F1 0.8172** (puncak 0.9101 pada Fold 5) dan presisi makro **82.5%**.
* **Kecepatan**: Latensi inferensi ONNX sangat rendah (**0.576 ms**), menjadikannya pilihan ideal untuk *real-time processing* di perangkat seluler tanpa menguras daya baterai.
* **Catatan Integrasi**: Gunakan file `models/hype_2.0.onnx` bersama runtime ONNX Mobile. Ekspor TFLite saat ini memicu operator kustom `ONNX_GEMM` pada unrolled loop CfC yang belum didukung oleh delegasi TFLite standar.


2. **HyperGrid v1.2 (Fallback Berdaya Rendah)**:
* **Ukuran**: File model ONNX paling ringkas (**70.6 KB**).
* **Fitur**: Memiliki segmentasi probabilitas 3 tingkat (*High*: 94, *Medium*: 149, *Low*: 80).
* **Peran**: Sangat cocok dijadikan sistem *fallback* pada arsitektur perangkat keras dengan keterbatasan RAM ekstrem.


3. **HyperGrid v1.0 (Arsip Baseline)**:
* Model dasar biner berbasis ekstraksi fitur jendela geser 5 Hz. Rentan terhadap alarm palsu (presisi hanya 45%) dan variasi posisi peletakan smartphone.



---

## Menjalankan Pipeline & Ekspor

### Pelatihan Model

Pilih skrip autorun sesuai arsitektur yang ingin dieksekusi:

```bash
# Model v1.0 (XGBoost Baseline)
uv run hype/autorun_hype_series_v1.py

# Model v1.2 (Causal Decomposition & Confidence Tiers)
uv run hype/autorun_hype_series_v2.py

# Model v2.0 (Liquid Neural Network CfC)
uv run hype/autorun_hype_series_2.0.py

```

### Ekspor & Konversi Edge Runtime

```bash
# Konversi checkpoint v2.0 ke ONNX dan varian TFLite
uv run hype/export_tflite_v2.py

```

Direktori keluaran:

* `models/hype_2.0.pt` (PyTorch Weights)
* `models/hype_2.0.onnx` (ONNX Runtime, 204.1 KB)
* `models/hype_liquid_v3_float32.tflite` (TFLite FP32)
* `models/hype_liquid_v3_float16.tflite` (TFLite FP16)
* `models/hype_scaler_v3.json` (Parameter normalisasi input)

---

## Eksplorasi Data & GIS Notebooks

Jalankan Jupyter Lab untuk menganalisis data telemetri dan memetakan sebaran lubang pada peta GIS interaktif:

```bash
uv run jupyter lab labs/

```

Daftar modul notebook:

* `lab_hype_1.1.ipynb`: Analisis eksplorasi fitur statistik dan evaluasi baseline v1.0.
* `lab_hype_1.2.ipynb`: Pengujian dekomposisi gravitasi vertikal/horizontal dan analisis Confidence Tiers.
* `lab_hype_2.0.ipynb`: Validasi kontinuitas waktu, variasi $\Delta t$, visualisasi temporal attention pooling, dan benchmarking CfCLiquidNet.

---