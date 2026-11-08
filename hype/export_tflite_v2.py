import os
import sys
import numpy as np

# Tambahkan root folder ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import load_raw_trips_v2
from utils.features import RoadFeatureExtractorV2

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow tidak ditemukan. Harap install TensorFlow di environment ini.")
    sys.exit(1)

def main():
    print("=" * 65)
    print("  HYPERGRID TFLITE EXPORTER (PROXY NEURAL NETWORK V2)")
    print("=" * 65)

    print("\n[Step 1/3] Memuat data dan mengekstrak fitur V2...")
    trips = load_raw_trips_v2(data_dir="./data", pothole_tolerance_sec=0.6, verbose=False)
    extractor = RoadFeatureExtractorV2(window_size=15, step_size=3)
    X, y, groups, metadata = extractor.extract_from_trips(trips)
    
    print(f" -> Matriks Fitur X : {X.shape}")
    print(f" -> Label Target y  : {y.shape} (Pothole: {(y==1).sum()}, Normal: {(y==0).sum()})")
    
    from sklearn.model_selection import train_test_split
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n[Step 2/3] Membangun dan melatih proxy Keras MLP dengan Normalization bawaan...")
    
    # Membuat layer Normalisasi TF
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))
    
    # Membangun model sederhana (MLP) yang akan dikonversi ke TFLite tanpa custom ops
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
        normalizer,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='roc_auc'), tf.keras.metrics.AUC(curve='PR', name='pr_auc')]
    )
    
    class_weight = {0: 1., 1: (y_train == 0).sum() / (y_train == 1).sum()}
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )
    
    loss, roc_auc, pr_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n -> Evaluasi Keras Model - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    
    print("\n[Step 3/3] Exporting ke TFLite...")
    # Convert ke TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimasi untuk ukuran model & inference cepat
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = "./models/hype_v2.tflite"
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"\n -> Model berhasil diekspor ke: {tflite_path}")
    print(f" -> Ukuran file: {os.path.getsize(tflite_path)/1024:.1f} KB")

if __name__ == "__main__":
    main()
