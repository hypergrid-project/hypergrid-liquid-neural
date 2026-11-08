import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import onnxruntime as rt
from utils.pull_data import pull_road_data
from utils.features import RoadFeatureExtractor, RoadFeatureExtractorV2
from utils.trainer import get_confidence_tier


def run_database_inference(version="v2", model_path=None):
    """
    Menjalankan simulasi inferensi real-time dari data sensor stream / database
    menggunakan model ONNX (V1 atau V2) dengan output Confidence score & Tier.
    """
    if model_path is None:
        model_path = f"./models/hype_v{version}.onnx"
        
    data = pull_road_data()
    
    # Fallback ke dataset lokal jika Supabase kosong atau error
    if not data:
        print("Data Supabase kosong / tidak dapat diakses. Menggunakan data sensor lokal sebagai sampel simulasi...")
        fallback_path = "./data/Pothole/trip1_sensors.csv"
        if not os.path.exists(fallback_path):
            fallback_path = os.path.join("..", fallback_path)
            
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path).head(200)
        else:
            print("Tidak ada data untuk diproses.")
            return
    else:
        df = pd.DataFrame(data)
        
    print(f"Data yang akan diproses: {len(df)} baris (Pipeline: {version.upper()})")

    # Mapping nama kolom jika format dari database berbeda
    if "sensor_x" in df.columns:
        df["accelerometerX"] = df["sensor_x"]
        df["accelerometerY"] = df["sensor_y"]
        df["accelerometerZ"] = df["sensor_z"]
    if "speed_kmh" in df.columns:
        df["speed"] = df["speed_kmh"]

    # Ekstraksi Fitur berdasarkan versi
    if version == "v2":
        extractor = RoadFeatureExtractorV2(window_size=15, step_size=3)
    else:
        extractor = RoadFeatureExtractor(window_size=20, step_size=5)
        
    X_stream, metadata = extractor.extract_from_stream(df)

    if X_stream.empty:
        print("Data tidak cukup untuk membentuk 1 window sensor.")
        return

    # Inisialisasi ONNX Runtime
    if not os.path.exists(model_path):
        alt_path = os.path.join("..", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"Model ONNX tidak ditemukan di {model_path}")
            return

    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    prob_name = session.get_outputs()[1].name

    inputs = X_stream.values.astype(np.float32)
    preds, probs = session.run([label_name, prob_name], {input_name: inputs})

    pothole_probs = probs[:, 1] if probs.ndim == 2 else probs
    confidence_scores = np.where(preds == 1, pothole_probs, 1.0 - pothole_probs)
    confidence_tiers = [get_confidence_tier(p) for p in pothole_probs]

    metadata["predicted_label"] = preds
    metadata["pothole_probability"] = pothole_probs
    metadata["confidence"] = confidence_scores
    metadata["confidence_tier"] = confidence_tiers
    metadata["status"] = np.where(preds == 1, "⚠️ POTHOLE", "✅ NORMAL")

    print(f"\n=== HASIL INFERENSI REAL-TIME ONNX ({version.upper()}) ===")
    print(metadata["status"].value_counts())
    print("\nRingkasan Confidence Tier:")
    print(metadata["confidence_tier"].value_counts())
    
    print("\nSampel Hasil Prediksi:")
    print(metadata[["timestamp", "latitude", "longitude", "peak_shock", "pothole_probability", "confidence", "confidence_tier", "status"]].head(10))


if __name__ == "__main__":
    # Jalankan inferensi default versi 2
    run_database_inference(version="v2")