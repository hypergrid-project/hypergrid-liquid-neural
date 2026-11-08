"""
HyperGrid Continuous-Time Deep Learning Pipeline
Component A: Data Pipeline & Preprocessing for Liquid Neural Networks (CfC)
=============================================================================
Features:
- Telemetry ingestion from CSV / Parquet streams (Accelerometer Ax, Ay, Az, Gyroscope Gx, Gy, Gz, Timestamp).
- Continuous-time Delta-t (dt) computation to model clock jitter and irregular sampling in CfC.
- Robust normalization (Median/IQR or Channel-wise Z-Score) invariant to sensor biases.
- Coordinate-invariance via L2 norms (||A||_2, ||G||_2) alongside 6-DoF raw channels.
- Realistic signal-level time-series augmentations:
  * Gaussian jittering
  * Suspension scaling
  * Time warping / velocity simulation
  * 3D spatial rotation around gravity axis
- Boundary-aware sliding window generator ensuring zero cross-trip data contamination.
- Full 4-Class Mapping:
  * Class 0: Normal Asphalt / Smooth Road
  * Class 1: Pothole / Severe Drop Impact
  * Class 2: Speed Bump / Asphalt Ridge
  * Class 3: Rough / Cobblestone / Uneven Pavement
"""

import os
import glob
import json
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class RobustSensorScaler:
    """
    Channel-wise robust normalization using median centering and Interquartile Range (IQR).
    Optionally supports standard Z-score (mean / std).
    Prevents sensor calibration biases across smartphone models and accelerometer hardware.
    """

    def __init__(self, method: str = "robust", eps: float = 1e-6):
        if method not in ["robust", "standard"]:
            raise ValueError(f"Method must be 'robust' or 'standard', got {method}")
        self.method = method
        self.eps = eps
        self.center: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "RobustSensorScaler":
        """
        Fit scaler parameters across all windows and time-steps.
        X shape: (N, sequence_length, channels) or (N, channels)
        """
        if X.ndim == 3:
            flat_X = X.reshape(-1, X.shape[-1])
        else:
            flat_X = X

        if self.method == "robust":
            q25 = np.percentile(flat_X, 25, axis=0)
            q50 = np.median(flat_X, axis=0)
            q75 = np.percentile(flat_X, 75, axis=0)
            iqr = q75 - q25
            iqr[iqr < self.eps] = 1.0

            self.center = q50.astype(np.float32)
            self.scale = iqr.astype(np.float32)
        else:
            mean = np.mean(flat_X, axis=0)
            std = np.std(flat_X, axis=0)
            std[std < self.eps] = 1.0

            self.center = mean.astype(np.float32)
            self.scale = std.astype(np.float32)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization to array X."""
        if not self.is_fitted:
            raise RuntimeError("RobustSensorScaler has not been fitted yet.")
        return (X - self.center) / self.scale

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse normalization on array X."""
        if not self.is_fitted:
            raise RuntimeError("RobustSensorScaler has not been fitted yet.")
        return (X * self.scale) + self.center

    def to_dict(self) -> Dict[str, list]:
        """Serialize parameters for edge deployment configuration."""
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted.")
        return {
            "method": self.method,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "RobustSensorScaler":
        """Load scaler from configuration dictionary."""
        scaler = cls(method=d["method"])
        scaler.center = np.array(d["center"], dtype=np.float32)
        scaler.scale = np.array(d["scale"], dtype=np.float32)
        scaler.is_fitted = True
        return scaler

    def save_json(self, file_path: str):
        """Save scaler parameters to JSON."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str) -> "RobustSensorScaler":
        """Load scaler parameters from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


# =============================================================================
# Signal-Level Data Augmentations for Time-Series
# =============================================================================

def augment_jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add zero-mean Gaussian noise to mimic high-frequency electrical sensor noise."""
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    return x + noise


def augment_scale(x: np.ndarray, min_scale: float = 0.85, max_scale: float = 1.15) -> np.ndarray:
    """Scale amplitude to simulate vehicle suspension stiffness or tire pressure variation."""
    factor = np.random.uniform(min_scale, max_scale)
    return x * factor


def augment_time_warp(x: np.ndarray, dt: np.ndarray, warp_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate vehicle acceleration or deceleration by warping time-steps via linear interpolation.
    Returns warped signal (x_warped) and adjusted delta-t (dt_warped).
    """
    seq_len, channels = x.shape
    warp_factor = np.random.uniform(warp_range[0], warp_range[1])
    orig_steps = np.linspace(0, 1, seq_len)
    
    # Warping curve
    new_steps = np.sort(np.random.uniform(0, 1, seq_len))
    new_steps = (new_steps - new_steps[0]) / (new_steps[-1] - new_steps[0] + 1e-7)

    x_warped = np.zeros_like(x)
    for c in range(channels):
        x_warped[:, c] = np.interp(new_steps, orig_steps, x[:, c])

    dt_warped = dt * warp_factor
    return x_warped.astype(np.float32), dt_warped.astype(np.float32)


def augment_spatial_rotation(x: np.ndarray, max_angle_deg: float = 20.0) -> np.ndarray:
    """
    Apply random 3D spatial rotation around the vertical gravity axis (or small random 3D tilt).
    Simulates smartphone mounting angle variations and cradle vibrations in a moving car.
    x columns are assumed to be [Ax, Ay, Az, Gx, Gy, Gz, ...]
    """
    theta = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotation matrix around Z axis
    R_z = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float32)

    x_rot = x.copy()
    # Rotate Accelerometer 3D vectors
    acc = x_rot[:, 0:3]
    x_rot[:, 0:3] = np.dot(acc, R_z.T)

    # Rotate Gyroscope 3D vectors
    if x.shape[1] >= 6:
        gyro = x_rot[:, 3:6]
        x_rot[:, 3:6] = np.dot(gyro, R_z.T)

    # If L2 norms were computed (channels 6 and 7), recompute them
    if x.shape[1] >= 8:
        x_rot[:, 6] = np.linalg.norm(x_rot[:, 0:3], axis=-1)
        x_rot[:, 7] = np.linalg.norm(x_rot[:, 3:6], axis=-1)

    return x_rot.astype(np.float32)


def apply_time_series_augmentations(
    x: np.ndarray,
    dt: np.ndarray,
    p_jitter: float = 0.5,
    p_scale: float = 0.5,
    p_warp: float = 0.3,
    p_rot: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies a composition of time-series augmentations stochastically."""
    x_aug = x.copy()
    dt_aug = dt.copy()

    if np.random.rand() < p_jitter:
        x_aug = augment_jitter(x_aug, sigma=0.02)
    if np.random.rand() < p_scale:
        x_aug = augment_scale(x_aug, min_scale=0.88, max_scale=1.12)
    if np.random.rand() < p_warp:
        x_aug, dt_aug = augment_time_warp(x_aug, dt_aug)
    if np.random.rand() < p_rot:
        x_aug = augment_spatial_rotation(x_aug, max_angle_deg=15.0)

    return x_aug, dt_aug


# =============================================================================
# Speed Bump & Ridge Synthesis (Class 2 Generator)
# =============================================================================

def synthesize_speed_bump_window(
    base_window: np.ndarray,
    dt: np.ndarray,
    bump_duration_sec: float = 0.6,
    nominal_dt: float = 0.02,
    vertical_axis: int = 1,
    pitch_axis: int = 3,
) -> np.ndarray:
    """
    Synthesize an authentic vehicle speed bump dynamic profile over a normal road window.
    Physics:
    - Entering speed bump produces positive vertical upward heave followed by suspension compression rebound.
    - Accompanying pitch rate oscillation in gyroscope.
    - Distinct from potholes which exhibit sudden drop negative acceleration followed by high-frequency impact ringing.
    """
    window = base_window.copy()
    seq_len = window.shape[0]

    # Sample position of speed bump center in window (middle third)
    center_idx = np.random.randint(seq_len // 3, 2 * seq_len // 3)
    steps_span = max(4, int(bump_duration_sec / nominal_dt))
    half_span = steps_span // 2

    start_idx = max(0, center_idx - half_span)
    end_idx = min(seq_len, center_idx + half_span)
    actual_len = end_idx - start_idx

    t = np.linspace(-np.pi, np.pi, actual_len)
    # Double-hump or damped sinusoidal heave profile
    heave_profile = (np.cos(t) + 1.0) * 0.5 * np.random.uniform(4.5, 9.0)  # Upward acceleration (m/s^2)
    # Pitch angular velocity profile (rad/s)
    pitch_profile = np.sin(t) * np.random.uniform(0.3, 0.8)

    window[start_idx:end_idx, vertical_axis] += heave_profile.astype(window.dtype)
    window[start_idx:end_idx, pitch_axis] += pitch_profile.astype(window.dtype)

    # Update L2 norms if present
    if window.shape[1] >= 8:
        window[:, 6] = np.linalg.norm(window[:, 0:3], axis=-1)
        window[:, 7] = np.linalg.norm(window[:, 3:6], axis=-1)

    return window


# =============================================================================
# Telemetry Stream Windowing & Dataset Loader
# =============================================================================

class TelemetryWindowExtractor:
    """
    Extracts sliding windows from multi-trip telemetry streams.
    Respects driving session/trip boundaries to avoid data leakage.
    Generates continuous-time delta-t timestamps for CfC recurrent dynamics.
    """

    def __init__(
        self,
        sequence_length: int = 100,
        stride: int = 50,
        nominal_hz: float = 50.0,
        include_magnitudes: bool = True,
        synthesize_bumps: bool = True,
        bump_ratio: float = 0.25,
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.nominal_dt = 1.0 / nominal_hz
        self.include_magnitudes = include_magnitudes
        self.synthesize_bumps = synthesize_bumps
        self.bump_ratio = bump_ratio

    def process_trip_df(
        self,
        df: pd.DataFrame,
        trip_id: str,
        road_class: Optional[int] = None,
        pothole_timestamps: Optional[np.ndarray] = None,
        pothole_tolerance_sec: float = 0.6,
    ) -> List[Dict[str, Union[np.ndarray, int, str]]]:
        """
        Extract windows from a single trip DataFrame.
        """
        df = df.copy()
        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
            ts = df["timestamp"].values
            dt = np.diff(ts, prepend=ts[0] + self.nominal_dt)
            # Clip delta-t to guard against clock resets or multi-hour sensor pauses
            dt = np.clip(dt, 0.001, 2.0).astype(np.float32)
        else:
            dt = np.full(len(df), fill_value=self.nominal_dt, dtype=np.float32)

        # Standardize column names
        sensor_mapping = {
            "accelerometerX": ["sensor_x", "ax", "acc_x", "Ax"],
            "accelerometerY": ["sensor_y", "ay", "acc_y", "Ay"],
            "accelerometerZ": ["sensor_z", "az", "acc_z", "Az"],
            "gyroX": ["sensor_gx", "gx", "gyro_x", "Gx"],
            "gyroY": ["sensor_gy", "gy", "gyro_y", "Gy"],
            "gyroZ": ["sensor_gz", "gz", "gyro_z", "Gz"],
        }
        for std_col, alts in sensor_mapping.items():
            if std_col not in df.columns:
                for alt in alts:
                    if alt in df.columns:
                        df[std_col] = df[alt]
                        break
                if std_col not in df.columns:
                    df[std_col] = 0.0

        raw_signals = df[
            ["accelerometerX", "accelerometerY", "accelerometerZ", "gyroX", "gyroY", "gyroZ"]
        ].values.astype(np.float32)

        # Clean NaN/Inf
        raw_signals = np.nan_to_num(raw_signals, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute L2 norms
        if self.include_magnitudes:
            acc_mag = np.linalg.norm(raw_signals[:, 0:3], axis=-1, keepdims=True)
            gyro_mag = np.linalg.norm(raw_signals[:, 3:6], axis=-1, keepdims=True)
            signals = np.concatenate([raw_signals, acc_mag, gyro_mag], axis=-1).astype(np.float32)
        else:
            signals = raw_signals

        # Generate pothole point-level mask if timestamps provided
        has_pothole_event = np.zeros(len(df), dtype=bool)
        if pothole_timestamps is not None and len(pothole_timestamps) > 0 and "timestamp" in df.columns:
            for pts in pothole_timestamps:
                mask = (df["timestamp"] >= (pts - pothole_tolerance_sec)) & (
                    df["timestamp"] <= (pts + pothole_tolerance_sec)
                )
                has_pothole_event[mask] = True

        windows = []
        n_samples = len(signals)

        if n_samples < self.sequence_length:
            # Pad short sequence if needed
            pad_len = self.sequence_length - n_samples
            pad_sig = np.pad(signals, ((0, pad_len), (0, 0)), mode="edge")
            pad_dt = np.pad(dt, (0, pad_len), mode="edge")
            label = 0 if road_class is None else road_class
            if np.any(has_pothole_event):
                label = 1
            windows.append({
                "signals": pad_sig,
                "dt": pad_dt.reshape(-1, 1),
                "label": label,
                "trip_id": trip_id,
            })
            return windows

        # Sliding window
        for start in range(0, n_samples - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            win_sig = signals[start:end]
            win_dt = dt[start:end].reshape(-1, 1)

            # Determine Class Target:
            # 1: Pothole, 2: Speed Bump, 3: Rough Road, 0: Smooth Road
            if np.any(has_pothole_event[start:end]):
                label = 1  # Pothole
            elif road_class is not None:
                label = road_class
            else:
                label = 0  # Normal smooth

            windows.append({
                "signals": win_sig,
                "dt": win_dt,
                "label": label,
                "trip_id": trip_id,
            })

            # Synthesize Speed Bump (Class 2) from normal smooth windows
            if self.synthesize_bumps and label == 0 and np.random.rand() < self.bump_ratio:
                bump_sig = synthesize_speed_bump_window(
                    win_sig,
                    win_dt,
                    bump_duration_sec=0.5,
                    nominal_dt=self.nominal_dt,
                )
                windows.append({
                    "signals": bump_sig,
                    "dt": win_dt.copy(),
                    "label": 2,  # Speed bump
                    "trip_id": trip_id,
                })

        return windows


def load_hypergrid_dataset(
    data_dir: str = "./data",
    sequence_length: int = 100,
    stride: int = 50,
    nominal_hz: float = 50.0,
    include_magnitudes: bool = True,
    synthesize_bumps: bool = True,
    bump_ratio: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads all trips from HyperGrid dataset folders (Pothole/ and RoadCondition/).
    Returns:
    - X: array of shape (N, sequence_length, channels)
    - dt: array of shape (N, sequence_length, 1)
    - y: array of shape (N,) class labels [0, 1, 2, 3]
    - groups: array of shape (N,) trip_ids for GroupKFold cross validation
    """
    extractor = TelemetryWindowExtractor(
        sequence_length=sequence_length,
        stride=stride,
        nominal_hz=nominal_hz,
        include_magnitudes=include_magnitudes,
        synthesize_bumps=synthesize_bumps,
        bump_ratio=bump_ratio,
    )

    all_windows = []

    # 1. Pothole Folder
    pothole_dir = os.path.join(data_dir, "Pothole")
    sensor_files = sorted(glob.glob(os.path.join(pothole_dir, "*_sensors.csv")))
    for sf in sensor_files:
        trip_name = os.path.basename(sf).replace("_sensors.csv", "")
        pf = os.path.join(pothole_dir, f"{trip_name}_potholes.csv")
        df_sensor = pd.read_csv(sf)
        pothole_ts = None
        if os.path.exists(pf):
            df_pot = pd.read_csv(pf)
            if "timestamp" in df_pot.columns and not df_pot.empty:
                pothole_ts = df_pot["timestamp"].values

        wins = extractor.process_trip_df(
            df=df_sensor,
            trip_id=f"pothole_{trip_name}",
            road_class=None,
            pothole_timestamps=pothole_ts,
        )
        all_windows.extend(wins)

    # 2. RoadCondition Folder: Good Roads (Class 0: Normal Asphalt)
    road_dir = os.path.join(data_dir, "RoadCondition")
    good_files = sorted(glob.glob(os.path.join(road_dir, "good*_sensors.csv")))
    for gf in good_files:
        trip_name = os.path.basename(gf).replace("_sensors.csv", "")
        df_good = pd.read_csv(gf)
        wins = extractor.process_trip_df(
            df=df_good,
            trip_id=f"road_{trip_name}",
            road_class=0,
        )
        all_windows.extend(wins)

    # 3. RoadCondition Folder: Bad Roads (Class 3: Rough / Cobblestone / Uneven)
    bad_files = sorted(glob.glob(os.path.join(road_dir, "bad*_sensors.csv")))
    for bf in bad_files:
        trip_name = os.path.basename(bf).replace("_sensors.csv", "")
        df_bad = pd.read_csv(bf)
        wins = extractor.process_trip_df(
            df=df_bad,
            trip_id=f"road_{trip_name}",
            road_class=3,
        )
        all_windows.extend(wins)

    if not all_windows:
        raise ValueError(f"No valid sensor CSV files found in {data_dir}")

    X = np.stack([w["signals"] for w in all_windows], axis=0).astype(np.float32)
    dt = np.stack([w["dt"] for w in all_windows], axis=0).astype(np.float32)
    y = np.array([w["label"] for w in all_windows], dtype=np.int64)
    groups = np.array([w["trip_id"] for w in all_windows], dtype=object)

    return X, dt, y, groups


# =============================================================================
# PyTorch Dataset Definition
# =============================================================================

class RoadSurfaceDataset(Dataset):
    """
    PyTorch Dataset serving (signals, dt, label) tuples.
    Applies optional dynamic data augmentations on-the-fly during training.
    """

    def __init__(
        self,
        X: np.ndarray,
        dt: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
    ):
        self.X = X.astype(np.float32)
        self.dt = dt.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_sample = self.X[idx]
        dt_sample = self.dt[idx]
        y_sample = self.y[idx]

        if self.augment:
            x_sample, dt_sample = apply_time_series_augmentations(
                x_sample, dt_sample, p_jitter=0.5, p_scale=0.5, p_warp=0.3, p_rot=0.4
            )

        return (
            torch.from_numpy(x_sample),
            torch.from_numpy(dt_sample),
            torch.tensor(y_sample, dtype=torch.long),
        )


def create_dataloader(
    X: np.ndarray,
    dt: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Factory helper to construct a PyTorch DataLoader."""
    dataset = RoadSurfaceDataset(X, dt, y, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


# =============================================================================
# Minimal Self-Test Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("   TESTING HYPERGRID LNN DATASET PIPELINE (dataset_lnn.py)")
    print("=" * 65)

    # 1. Test RobustSensorScaler
    dummy_signals = np.random.randn(10, 100, 8).astype(np.float32) * 5.0 + 2.0
    scaler = RobustSensorScaler(method="robust")
    scaler.fit(dummy_signals)
    transformed = scaler.transform(dummy_signals)
    reconstructed = scaler.inverse_transform(transformed)
    assert np.allclose(dummy_signals, reconstructed, atol=1e-5), "Scaler inverse transform failed!"
    print(f"[*] RobustSensorScaler sanity check passed: center={scaler.center.shape}, scale={scaler.scale.shape}")

    # 2. Test Time-Series Augmentations
    x_test = np.random.randn(100, 8).astype(np.float32)
    dt_test = np.ones((100, 1), dtype=np.float32) * 0.02
    x_aug, dt_aug = apply_time_series_augmentations(x_test, dt_test)
    assert x_aug.shape == x_test.shape, "Augmented shape mismatch!"
    assert dt_aug.shape == dt_test.shape, "Augmented dt shape mismatch!"
    print(f"[*] Time-series augmentations passed: {x_aug.shape}")

    # 3. Test Loading Real HyperGrid Data
    data_path = "./data"
    if os.path.exists(data_path):
        print(f"[*] Loading raw dataset from '{data_path}'...")
        X, dt, y, groups = load_hypergrid_dataset(
            data_dir=data_path,
            sequence_length=100,
            stride=25,
            nominal_hz=5.0,
            include_magnitudes=True,
            synthesize_bumps=True,
            bump_ratio=0.3,
        )
        print(f" -> X shape: {X.shape} (channels: 8 = 6-DoF + 2 Magnitudes)")
        print(f" -> dt shape: {dt.shape}")
        print(f" -> y shape: {y.shape} distribution: {np.bincount(y)}")
        print(f" -> Unique trips: {len(np.unique(groups))}")

        # Test DataLoader
        loader = create_dataloader(X, dt, y, batch_size=16, shuffle=True, augment=True)
        batch_x, batch_dt, batch_y = next(iter(loader))
        print(f" -> Batch x: {batch_x.shape}, batch_dt: {batch_dt.shape}, batch_y: {batch_y.shape}")
        print("[*] Dataset pipeline validation complete!")
    else:
        print("[!] Data directory not found, tested synthetic flow successfully.")
