import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from xgboost import XGBClassifier
import onnxruntime as rt
from onnxmltools import convert_xgboost as convert_xgb
from onnxmltools.convert.common.data_types import FloatTensorType as OFloatTensorType


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_confidence_tier(prob):
    """
    Mengelompokkan tingkat keyakinan model (Confidence Tier):
    - HIGH   : Probabilitas >= 0.80 (Sangat yakin lubang)
    - MEDIUM : Probabilitas 0.50 - 0.79 (Terdeteksi lubang, perlu agregasi)
    - LOW    : Probabilitas < 0.50 (Normal / jalan bergelombang biasa)
    """
    if prob >= 0.80:
        return "HIGH"
    elif prob >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def train_hypergrid_model(
    X,
    y,
    groups,
    metadata,
    model_output_path="./models/hype_v1.onnx",
    test_size=0.2,
    random_state=42,
    tune_hyperparameters=True,
):
    """
    Melatih model XGBoost HyperGrid v1 dengan Group-Aware Split (Trip-based) dan ekspor ke ONNX.
    """
    if len(X) == 0 or len(groups) == 0:
        raise ValueError("Matriks fitur X atau groups kosong (0 sampel).")

    if not os.path.isabs(model_output_path):
        model_output_path = os.path.join(PROJECT_ROOT, model_output_path.lstrip("./"))

    print("=" * 65)
    print("      HYPERGRID MACHINE LEARNING PIPELINE TRAINING (V1)")
    print("=" * 65)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train, groups_test = groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy()
    meta_train, meta_test = metadata.iloc[train_idx].copy(), metadata.iloc[test_idx].copy()
    
    print(f"\n1. DATA SPLIT (GROUP-AWARE):")
    print(f" - Train Samples: {len(X_train)} (from {groups_train.nunique()} trips: {list(groups_train.unique())})")
    print(f" - Test Samples : {len(X_test)} (from {groups_test.nunique()} trips: {list(groups_test.unique())})")
    print(f" - Train Pothole Ratio: {(y_train == 1).mean():.1%}")
    print(f" - Test Pothole Ratio : {(y_test == 1).mean():.1%}")
    
    overlap = set(groups_train).intersection(set(groups_test))
    assert len(overlap) == 0, f"Error: Data leakage detected on trips: {overlap}"
    
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f" - XGBoost scale_pos_weight: {scale_pos:.2f}")
    
    base_xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        n_jobs=-1,
    )
    
    if tune_hyperparameters:
        print("\n2. HYPERPARAMETER TUNING VIA GROUP-K-FOLD...")
        gkf = GroupKFold(n_splits=min(5, groups_train.nunique()))
        param_dist = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1, 0.15],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "gamma": [0, 0.1, 0.2],
        }
        
        cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1",
            cv=cv_splits,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f" - Best CV F1-Score: {search.best_score_:.4f}")
        print(f" - Best Parameters : {search.best_params_}")
    else:
        best_model = base_xgb
        best_model.fit(X_train, y_train)
        
    print("\n3. EVALUASI TEST SET (UNSEEN TRIPS):")
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_thresh_idx = np.argmax(f1_scores)
    optimal_thresh = float(thresholds[best_thresh_idx]) if best_thresh_idx < len(thresholds) else 0.5
    optimal_thresh = np.clip(optimal_thresh, 0.3, 0.7)
    
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    y_test_pred_opt = (y_test_proba >= optimal_thresh).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f" - ROC-AUC Score: {roc_auc:.4f}")
    print(f" - Optimal Threshold (F1-Tuned): {optimal_thresh:.3f}")
    
    print("\n--- Classification Report (Default Threshold 0.5) ---")
    print(classification_report(y_test, y_test_pred_default, target_names=["Normal", "Pothole"]))
    
    print(f"--- Classification Report (Optimal Threshold {optimal_thresh:.3f}) ---")
    print(classification_report(y_test, y_test_pred_opt, target_names=["Normal", "Pothole"]))
    
    cm = confusion_matrix(y_test, y_test_pred_opt)
    print("Confusion Matrix:")
    print(f" TN: {cm[0, 0]} | FP: {cm[0, 1]}")
    print(f" FN: {cm[1, 0]} | TP: {cm[1, 1]}")
    
    # Export ONNX
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    onnx_ready_model = XGBClassifier(**best_model.get_params())
    onnx_ready_model.fit(X_train.values, y_train.values)
    
    num_features = X_train.shape[1]
    initial_types = [("float_input", OFloatTensorType([None, num_features]))]
    onnx_model = convert_xgb(onnx_ready_model, initial_types=initial_types, target_opset=12)
    
    with open(model_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f" - Model ONNX berhasil disimpan ke: {model_output_path}")
    
    df_results = meta_test.copy()
    df_results["actual_label"] = y_test.values
    df_results["predicted_label"] = y_test_pred_opt
    df_results["pothole_probability"] = y_test_proba
    df_results["confidence"] = np.where(y_test_pred_opt == 1, y_test_proba, 1.0 - y_test_proba)
    df_results["confidence_tier"] = df_results["pothole_probability"].apply(get_confidence_tier)
    
    return {
        "model": best_model,
        "optimal_threshold": optimal_thresh,
        "roc_auc": roc_auc,
        "feature_names": list(X.columns),
        "results_df": df_results,
    }


def train_hypergrid_model_v2(
    X,
    y,
    groups,
    metadata,
    model_output_path="./models/hype_v2.onnx",
    test_size=0.2,
    random_state=42,
    tune_hyperparameters=True,
):
    """
    Melatih model XGBoost HyperGrid v2 dengan pembaruan:
    - StratifiedGroupKFold untuk cross-validation seimbang (tanpa empty positive folds).
    - Skor Confidence & Confidence Tiers (HIGH, MEDIUM, LOW).
    - Optimasi PR-AUC (Average Precision) & ROC-AUC.
    - Ekspor ONNX mandiri dan verifikasi runtime.
    """
    if len(X) == 0 or len(groups) == 0:
        raise ValueError("Matriks fitur X atau groups kosong (0 sampel).")

    if not os.path.isabs(model_output_path):
        model_output_path = os.path.join(PROJECT_ROOT, model_output_path.lstrip("./"))

    print("=" * 65)
    print("      HYPERGRID MACHINE LEARNING PIPELINE TRAINING (V2)")
    print("=" * 65)
    
    # 1. Stratified Group Splitting untuk Test Set
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train, groups_test = groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy()
    meta_train, meta_test = metadata.iloc[train_idx].copy(), metadata.iloc[test_idx].copy()
    
    print(f"\n1. DATA SPLIT (GROUP-AWARE V2):")
    print(f" - Train Samples: {len(X_train)} (from {groups_train.nunique()} trips: {list(groups_train.unique())})")
    print(f" - Test Samples : {len(X_test)} (from {groups_test.nunique()} trips: {list(groups_test.unique())})")
    print(f" - Train Pothole Ratio: {(y_train == 1).mean():.1%} ({(y_train == 1).sum()} lubang)")
    print(f" - Test Pothole Ratio : {(y_test == 1).mean():.1%} ({(y_test == 1).sum()} lubang)")
    
    overlap = set(groups_train).intersection(set(groups_test))
    assert len(overlap) == 0, f"Error: Data leakage detected on trips: {overlap}"
    
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f" - XGBoost scale_pos_weight: {scale_pos:.2f}")
    
    # 2. Hyperparameter Search dengan StratifiedGroupKFold
    base_xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=scale_pos * 0.85,  # Moderasi weight untuk kalibrasi probabilitas yang lebih baik
        n_jobs=-1,
    )
    
    if tune_hyperparameters:
        print("\n2. HYPERPARAMETER TUNING VIA STRATIFIED GROUP-K-FOLD...")
        # Hitung jumlah trip yang memiliki sampel lubang di data latih
        positive_groups_count = groups_train[y_train == 1].nunique()
        n_splits = max(2, min(3, positive_groups_count))
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        
        param_dist = {
            "n_estimators": [60, 100, 150, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.08, 0.12],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.75, 0.85, 1.0],
            "gamma": [0, 0.1, 0.2],
            "min_child_weight": [1, 3, 5],
        }
        
        cv_splits = list(sgkf.split(X_train, y_train, groups=groups_train))
        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=15,
            scoring="roc_auc",
            cv=cv_splits,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f" - Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f" - Best Parameters: {search.best_params_}")
    else:
        best_model = base_xgb
        best_model.fit(X_train, y_train)
        
    # 3. Evaluasi terhadap Test Set
    print("\n3. EVALUASI TEST SET (UNSEEN TRIPS V2):")
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_thresh_idx = np.argmax(f1_scores)
    optimal_thresh = float(thresholds[best_thresh_idx]) if best_thresh_idx < len(thresholds) else 0.5
    optimal_thresh = float(np.clip(optimal_thresh, 0.25, 0.75))
    
    y_test_pred_opt = (y_test_proba >= optimal_thresh).astype(int)
    
    print(f" - ROC-AUC Score: {roc_auc:.4f}")
    print(f" - PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f" - Optimal Decision Threshold: {optimal_thresh:.3f}")
    
    print(f"\n--- Classification Report (Optimal Threshold {optimal_thresh:.3f}) ---")
    print(classification_report(y_test, y_test_pred_opt, target_names=["Normal", "Pothole"]))
    
    cm = confusion_matrix(y_test, y_test_pred_opt)
    print("Confusion Matrix:")
    print(f" TN: {cm[0, 0]} | FP: {cm[0, 1]}")
    print(f" FN: {cm[1, 0]} | TP: {cm[1, 1]}")
    
    # 4. Confidence Analysis
    confidence_scores = np.where(y_test_pred_opt == 1, y_test_proba, 1.0 - y_test_proba)
    confidence_tiers = [get_confidence_tier(p) for p in y_test_proba]
    
    print("\n--- Distribusi Tingkat Confidence pada Prediksi Lubang ---")
    pothole_indices = np.where(y_test_pred_opt == 1)[0]
    if len(pothole_indices) > 0:
        high_conf = sum(1 for i in pothole_indices if confidence_tiers[i] == "HIGH")
        med_conf = sum(1 for i in pothole_indices if confidence_tiers[i] == "MEDIUM")
        low_conf = sum(1 for i in pothole_indices if confidence_tiers[i] == "LOW")
        print(f" - High Confidence   (>=80%): {high_conf} ({high_conf/len(pothole_indices):.1%})")
        print(f" - Medium Confidence (50-79%): {med_conf} ({med_conf/len(pothole_indices):.1%})")
        print(f" - Low / Ambiguous   (<50%):  {low_conf} ({low_conf/len(pothole_indices):.1%})")
    
    # 5. Export ke ONNX
    print("\n4. EXPORT MODEL V2 KE ONNX...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    onnx_ready_model = XGBClassifier(**best_model.get_params())
    onnx_ready_model.fit(X_train.values, y_train.values)
    
    num_features = X_train.shape[1]
    initial_types = [("float_input", OFloatTensorType([None, num_features]))]
    
    onnx_model = convert_xgb(
        onnx_ready_model,
        initial_types=initial_types,
        target_opset=12,
    )
    
    with open(model_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f" - Model ONNX V2 berhasil disimpan ke: {model_output_path}")
    print(f" - Ukuran File: {os.path.getsize(model_output_path) / 1024:.1f} KB")
    print(f" - Jumlah Fitur Input: {num_features}")
    
    # 6. Verifikasi ONNX Runtime
    print("\n5. VERIFIKASI ONNX RUNTIME V2...")
    sess = rt.InferenceSession(model_output_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    prob_name = sess.get_outputs()[1].name
    
    test_batch = X_test.iloc[:10].values.astype(np.float32)
    onnx_pred, onnx_prob = sess.run([label_name, prob_name], {input_name: test_batch})
    py_pred = onnx_ready_model.predict(test_batch)
    
    is_matching = np.array_equal(py_pred, onnx_pred)
    print(f" - Verifikasi Hasil Python vs ONNX: {'SUKSES (IDENTIK)' if is_matching else 'PERBEDAAN TERDETEKSI'}")
    
    df_results = meta_test.copy()
    df_results["actual_label"] = y_test.values
    df_results["predicted_label"] = y_test_pred_opt
    df_results["pothole_probability"] = y_test_proba
    df_results["confidence"] = confidence_scores
    df_results["confidence_tier"] = confidence_tiers
    
    return {
        "model": best_model,
        "optimal_threshold": optimal_thresh,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "feature_names": list(X.columns),
        "results_df": df_results,
    }
