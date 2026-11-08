"""
HyperGrid Series v2 - Automated End-to-End Pipeline
===================================================
Script ini menjalankan seluruh siklus pemrosesan data V2:
1. Pembersihan data sensor (interpolasi GPS negative speed, penanganan noise, sekuensial timestamp).
2. Ekstraksi fitur invarian orientasi kausal & dekomposisi getaran vertikal/horizontal (RoadFeatureExtractorV2).
3. Training model XGBoost berbasis Stratified GroupKFold cross-validation.
4. Perhitungan skor Confidence dan Confidence Tiers (HIGH, MEDIUM, LOW).
5. Ekspor model ke format ONNX (models/hype_v2.onnx) dan verifikasi runtime.
"""

import os
import sys

# Tambahkan root folder ke sys.path agar import utils selalu bekerja
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import load_raw_trips_v2
from utils.features import RoadFeatureExtractorV2
from utils.trainer import train_hypergrid_model_v2


def main():
    print("=" * 65)
    print("      HYPERGRID MACHINE LEARNING PIPELINE (AUTORUN V2)")
    print("=" * 65)

    # 1. Load Raw Trips dengan Data Cleaning V2
    print("\n[Step 1/4] Memuat data perjalanan dan membersihkan anomali sensor (V2)...")
    trips = load_raw_trips_v2(data_dir="./data", pothole_tolerance_sec=0.6, verbose=True)

    # 2. Ekstraksi Fitur Kausal & Dekomposisi Vertikal/Horizontal (Sliding Window V2)
    print("\n[Step 2/4] Mengekstrak fitur dekomposisi vertikal/horizontal dan dinamika sinyal...")
    extractor = RoadFeatureExtractorV2(window_size=15, step_size=3)
    X, y, groups, metadata = extractor.extract_from_trips(trips)

    print(f" -> Matriks Fitur X : {X.shape}")
    print(f" -> Label Target y  : {y.shape} (Pothole: {(y==1).sum()}, Normal: {(y==0).sum()})")
    print(f" -> Jumlah Sesi Trip: {groups.nunique()}")

    # 3. Training & Evaluasi Model V2
    print("\n[Step 3/4] Melatih model dengan Stratified GroupKFold & Evaluasi Unseen Trips...")
    model_path = "./models/hype_v2.onnx"
    results = train_hypergrid_model_v2(
        X=X,
        y=y,
        groups=groups,
        metadata=metadata,
        model_output_path=model_path,
        test_size=0.2,
        random_state=42,
        tune_hyperparameters=True,
    )

    # 4. Ringkasan Pipeline V2
    res_df = results["results_df"]
    potholes_detected = (res_df["predicted_label"] == 1).sum()
    high_conf_potholes = (res_df["confidence_tier"] == "HIGH").sum()
    
    print("\n[Step 4/4] Pipeline V2 selesai dijalankan!")
    print(f" -> Model ONNX V2 tersimpan di : {model_path}")
    print(f" -> ROC-AUC Test Score         : {results['roc_auc']:.4f}")
    print(f" -> PR-AUC (Avg Precision)     : {results['pr_auc']:.4f}")
    print(f" -> Ambang Batas Optimal       : {results['optimal_threshold']:.3f}")
    print(f" -> Total Pothole Terdeteksi   : {potholes_detected} titik")
    print(f" -> High-Confidence Potholes   : {high_conf_potholes} titik")
    print("=" * 65)


if __name__ == "__main__":
    main()
