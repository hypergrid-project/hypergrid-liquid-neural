"""
HyperGrid Series v3 - Automated End-to-End Liquid Neural Network (CfC) Pipeline
================================================================================
Script ini menjalankan seluruh siklus pemrosesan data & training V3:
1. Pembersihan data sensor telemetri mentah (Ax, Ay, Az, Gx, Gy, Gz) dan isolasi inter-sample delta-t (dt).
2. Perhitungan magnitudo invarian orientasi (||A||_2, ||G||_2) dan Robust Normalization (Median/IQR).
3. Ekstraksi sekuens temporal (sliding window: 100 step, stride: 30) dengan isolasi sesi perjalanan (Group-Aware).
4. Training Closed-form Continuous-time Network (CfCLiquidNet) dengan:
   - 1D Depthwise-Separable Convolutional front-end (ekstraksi micro-vibration shock).
   - Core Recurrent CfC Layer (continuous-depth analytic solution tanpa numerical ODE solver).
   - Learned Temporal Attention Pooling & Linear Classification Head.
   - Multi-Class Focal Loss & AdamW Cosine Annealing.
5. Evaluasi K-Fold Stratified Group Cross-Validation (mencegah data leakage antar segmen jalan identik).
6. Ekspor Model Siap Edge:
   - PyTorch Checkpoint: models/hype_2.0.pt
   - ONNX Runtime: models/hype_2.0.onnx
   - TensorFlow Lite: models/hype_2.0.tflite (Float32, Float16, INT8)
   - Scaler Config: models/hype_scaler_v3.json
7. Verifikasi Paritas Numerik (PyTorch vs ONNX vs TFLite) & Pengukuran Latensi Edge Inference.
"""

import os
import sys
import time
import json
import numpy as np
import torch

# Tambahkan root folder ke sys.path agar import utils / modul root selalu bekerja
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.dataset_lnn import load_hypergrid_dataset, RobustSensorScaler, create_dataloader
from utils.model_cfc import build_cfc_model, CfCLiquidNet
from utils.train_cfc import train_cfc_pipeline, evaluate, compute_telemetry_metrics
from utils.export_edge import export_model_to_onnx, export_onnx_to_tflite, verify_numerical_parity, benchmark_inference_latency


def main():
    print("=" * 70)
    print("     HYPERGRID MACHINE LEARNING PIPELINE (AUTORUN LIQUID V3)")
    print("  Closed-form Continuous-Time Neural Networks (CfC) for Edge Telemetry")
    print("=" * 70)

    data_dir = os.path.join(ROOT_DIR, "data")
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    sequence_length = 100
    stride = 30
    in_channels = 8  # 6-DoF + 2 Magnitudes
    num_classes = 4  # 0: Smooth, 1: Pothole, 2: Bump, 3: Rough
    hidden_size = 32
    epochs = 20
    batch_size = 32
    lr = 1e-3

    # 1. Ingest Data Mentah dan Ekstraksi Sekuens Kontinu
    print("\n[Step 1/5] Memuat telemetri sensor multi-trip & ekstraksi sekuens continuous-time...")
    X, dt, y, groups = load_hypergrid_dataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        stride=stride,
        nominal_hz=5.0,
        include_magnitudes=True,
        synthesize_bumps=True,
        bump_ratio=0.25,
    )

    print(f" -> Matriks Sinyal X : {X.shape} (N, Length, Channels)")
    print(f" -> Vektor Delta-t dt: {dt.shape} (Rentang dt: {dt.min():.4f}s s/d {dt.max():.4f}s)")
    print(f" -> Distribusi Label y: {np.bincount(y)}")
    print(f"    * Class 0 (Smooth Normal) : {(y == 0).sum()} windows")
    print(f"    * Class 1 (Pothole Drop)  : {(y == 1).sum()} windows")
    print(f"    * Class 2 (Speed Bump)    : {(y == 2).sum()} windows")
    print(f"    * Class 3 (Rough Surface) : {(y == 3).sum()} windows")
    print(f" -> Jumlah Trip Terisolasi: {len(np.unique(groups))} sesi perjalanan unik")

    # 2. Training dengan Stratified GroupKFold & Multi-Class Focal Loss
    print("\n[Step 2/5] Melatih model CfC Liquid Net dengan Group-Aware Cross-Validation...")
    cfc_model_path = os.path.join(models_dir, "hype_2.0.pt")
    scaler_path = os.path.join(models_dir, "hype_scaler_v3.json")

    training_results = train_cfc_pipeline(
        data_dir=data_dir,
        output_dir=os.path.join(models_dir, "cfc"),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-4,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        stride=stride,
        loss_type="focal",
        k_folds=5,
        device_name="auto",
        use_conv_frontend=True,
    )

    best_pt_path = training_results["best_model_path"]
    best_scaler_path = training_results["scaler_path"]

    # Salin best model ke lokasi standar models/
    if os.path.exists(best_pt_path):
        import shutil
        shutil.copyfile(best_pt_path, cfc_model_path)
    if os.path.exists(best_scaler_path):
        import shutil
        shutil.copyfile(best_scaler_path, scaler_path)

    # 3. Evaluasi Metrik Produksi & Validasi Unseen Trips
    print("\n[Step 3/5] Ringkasan Metrik Evaluasi Model V3:")
    print(f" -> Validation Macro F1-Score: {training_results.get('mean_macro_f1', 0.0):.4f}")
    print(f" -> Validation Macro Precision: {training_results.get('mean_macro_precision', 0.0):.4f}")
    print(f" -> Validation Macro Recall   : {training_results.get('mean_macro_recall', 0.0):.4f}")
    print(f" -> Normal Road FPR (False Alarm): {training_results.get('mean_normal_road_fpr', 0.0):.4%}")
    print(f" -> Best Overall F1-Score    : {training_results.get('best_overall_f1', 0.0):.4f}")
    print(f" -> Checkpoint PyTorch tersimpan di: {cfc_model_path}")

    # 4. Ekspor ke ONNX & TensorFlow Lite Edge Formats
    print("\n[Step 4/5] Mengekspor model ke format ONNX dan TensorFlow Lite (Float32, Float16, INT8)...")
    model_eval = build_cfc_model(
        in_channels=in_channels,
        hidden_size=hidden_size,
        num_classes=num_classes,
        use_conv_frontend=True,
        backbone_type="analytic",
    )
    checkpoint = torch.load(cfc_model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_eval.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_eval.load_state_dict(checkpoint)
    model_eval.eval()

    onnx_path = os.path.join(models_dir, "hype_2.0.onnx")
    export_model_to_onnx(
        model=model_eval,
        output_path=onnx_path,
        sequence_length=sequence_length,
        in_channels=in_channels,
        opset_version=16,
    )

    # Export TFLite
    try:
        tflite_dict = export_onnx_to_tflite(
            onnx_path=onnx_path,
            output_dir=models_dir,
            calibration_data=(X[:200], dt[:200]),
        )
    except Exception as e:
        print(f" -> [!] Konversi TFLite dilewati atau terjadi kendala: {e}")
        tflite_dict = {}

    # 5. Paritas Numerik & Benchmarking Latensi CPU
    print("\n[Step 5/5] Menguji Paritas Numerik & Benchmarking Latensi Inferensi...")
    tflite_f32_path = tflite_dict.get("float32", os.path.join(models_dir, "hype_2.0.tflite"))
    parity = verify_numerical_parity(
        torch_model=model_eval,
        onnx_path=onnx_path,
        tflite_path=tflite_f32_path if os.path.exists(tflite_f32_path) else None,
        num_test_windows=500,
        sequence_length=sequence_length,
        in_channels=in_channels,
    )

    print("\n=== HASIL BENCHMARK LATENSI INFERENSI (CPU) ===")
    onnx_bench = benchmark_inference_latency(onnx_path, engine="onnx", num_cycles=1000)
    if os.path.exists(tflite_f32_path):
        tflite_bench = benchmark_inference_latency(tflite_f32_path, engine="tflite", num_cycles=1000)
    else:
        tflite_bench = {"mean_ms": 0.0}

    print("\n" + "=" * 70)
    print("       HYPERGRID LIQUID NEURAL NETWORK V3 SELESAI DIEKSEKUSI!")
    print("=" * 70)
    print(f" -> PyTorch Checkpoint  : {cfc_model_path}")
    print(f" -> ONNX Model          : {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    if os.path.exists(tflite_f32_path):
        print(f" -> TFLite Float32      : {tflite_f32_path} ({os.path.getsize(tflite_f32_path)/1024:.1f} KB)")
    if "float16" in tflite_dict and os.path.exists(tflite_dict["float16"]):
        print(f" -> TFLite Float16      : {tflite_dict['float16']} ({os.path.getsize(tflite_dict['float16'])/1024:.1f} KB)")
    if "int8" in tflite_dict and os.path.exists(tflite_dict["int8"]):
        print(f" -> TFLite INT8 Quant   : {tflite_dict['int8']} ({os.path.getsize(tflite_dict['int8'])/1024:.1f} KB)")
    print(f" -> Scaler Configuration: {scaler_path}")
    print(f" -> Latensi Inferensi   : ONNX={onnx_bench['mean_ms']:.3f} ms | TFLite={tflite_bench.get('mean_ms', 0.0):.3f} ms")
    print(f" -> F1-Score Validasi   : {training_results.get('mean_macro_f1', 0.0):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
