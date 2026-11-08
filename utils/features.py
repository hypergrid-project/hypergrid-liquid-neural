import numpy as np
import pandas as pd


class RoadFeatureExtractor:
    """
    Ekstraktor Fitur Sensor Jalan (HyperGrid) v1 - Invarian orientasi berbasis
    rolling median baseline.
    """
    def __init__(self, window_size=20, step_size=5, g_constant=1.0):
        self.window_size = window_size
        self.step_size = step_size
        self.g_constant = g_constant
        
    def _preprocess_trip_signals(self, df_trip):
        df = df_trip.copy()
        
        ax = df["accelerometerX"] if "accelerometerX" in df.columns else df.get("sensor_x", 0.0)
        ay = df["accelerometerY"] if "accelerometerY" in df.columns else df.get("sensor_y", 0.0)
        az = df["accelerometerZ"] if "accelerometerZ" in df.columns else df.get("sensor_z", 0.0)
        
        gx = df["gyroX"] if "gyroX" in df.columns else df.get("sensor_gx", 0.0)
        gy = df["gyroY"] if "gyroY" in df.columns else df.get("sensor_gy", 0.0)
        gz = df["gyroZ"] if "gyroZ" in df.columns else df.get("sensor_gz", 0.0)
        
        speed = df["speed"] if "speed" in df.columns else df.get("speed_kmh", 0.0)
        
        df["acc_mag"] = np.sqrt(ax**2 + ay**2 + az**2)
        df["gyro_mag"] = np.sqrt(gx**2 + gy**2 + gz**2)
        
        baseline = df["acc_mag"].rolling(window=max(self.window_size * 2, 30), min_periods=1, center=True).median()
        df["dynamic_acc"] = (df["acc_mag"] - baseline).abs()
        
        df["jerk"] = df["dynamic_acc"].diff().abs().fillna(0.0)
        df["angular_jerk"] = df["gyro_mag"].diff().abs().fillna(0.0)
        df["speed_diff"] = speed.diff().abs().fillna(0.0)
        df["speed_clean"] = speed
        
        return df

    def _extract_window_features(self, window_df):
        features = {}
        target_signals = ["dynamic_acc", "gyro_mag", "jerk", "angular_jerk", "speed_clean"]
        
        for sig in target_signals:
            vals = window_df[sig].values
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            rng_val = max_val - min_val
            rms_val = np.sqrt(np.mean(vals**2))
            
            features[f"{sig}_mean"] = float(mean_val)
            features[f"{sig}_std"] = float(std_val)
            features[f"{sig}_min"] = float(min_val)
            features[f"{sig}_max"] = float(max_val)
            features[f"{sig}_range"] = float(rng_val)
            features[f"{sig}_rms"] = float(rms_val)
            
        dyn_vals = window_df["dynamic_acc"].values
        rms_dyn = features["dynamic_acc_rms"]
        max_dyn = features["dynamic_acc_max"]
        
        features["dynamic_acc_crest_factor"] = float(max_dyn / (rms_dyn + 1e-6))
        centered_dyn = dyn_vals - np.median(dyn_vals)
        zcr = ((centered_dyn[:-1] * centered_dyn[1:]) < 0).sum() / (len(dyn_vals) - 1 + 1e-6)
        features["dynamic_acc_zcr"] = float(zcr)
        
        return features

    def extract_from_trips(self, trips_list):
        records = []
        labels = []
        groups = []
        meta_list = []
        
        for trip_df in trips_list:
            if len(trip_df) < self.window_size:
                continue
                
            trip_id = trip_df["trip_id"].iloc[0] if "trip_id" in trip_df.columns else "unknown"
            df_processed = self._preprocess_trip_signals(trip_df)
            
            n_samples = len(df_processed)
            for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                win = df_processed.iloc[start_idx:end_idx]
                
                feat = self._extract_window_features(win)
                records.append(feat)
                
                win_label = 1 if (win["label"] == 1).any() else 0
                labels.append(win_label)
                groups.append(trip_id)
                
                meta_list.append({
                    "trip_id": trip_id,
                    "timestamp": win["timestamp"].iloc[-1] if "timestamp" in win.columns else 0,
                    "latitude": win["latitude"].mean() if "latitude" in win.columns else 0.0,
                    "longitude": win["longitude"].mean() if "longitude" in win.columns else 0.0,
                    "peak_shock": feat["dynamic_acc_max"],
                    "speed": feat["speed_clean_mean"],
                })
                
        X = pd.DataFrame(records)
        y = pd.Series(labels, name="label")
        groups = pd.Series(groups, name="trip_id")
        metadata = pd.DataFrame(meta_list)
        
        return X, y, groups, metadata

    def extract_from_stream(self, df_stream):
        if len(df_stream) < self.window_size:
            return pd.DataFrame(), pd.DataFrame()
            
        df_processed = self._preprocess_trip_signals(df_stream)
        records = []
        meta_list = []
        
        n_samples = len(df_processed)
        for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            win = df_processed.iloc[start_idx:end_idx]
            
            feat = self._extract_window_features(win)
            records.append(feat)
            
            meta_list.append({
                "timestamp": win["timestamp"].iloc[-1] if "timestamp" in win.columns else 0,
                "latitude": win["latitude"].mean() if "latitude" in win.columns else 0.0,
                "longitude": win["longitude"].mean() if "longitude" in win.columns else 0.0,
                "peak_shock": feat["dynamic_acc_max"],
                "speed": feat["speed_clean_mean"],
            })
            
        X = pd.DataFrame(records)
        metadata = pd.DataFrame(meta_list)
        return X, metadata


class RoadFeatureExtractorV2:
    """
    Ekstraktor Fitur Sensor Jalan (HyperGrid) v2:
    =============================================
    Pembaruan Utama:
    1. Causal Gravity Tracking (EMA): Menghilangkan train-serve skew.
    2. Vertical-Horizontal Decomposition: Memisahkan getaran vertikal lubang ($a_v$)
       dari manuver horizontal ($a_h$).
    3. Speed-Normalized Shock Intensity: Normalisasi dampak getaran terhadap kecepatan kendaraan.
    4. Causal Feature Engineering: Konsistensi 100% antara offline training dan real-time stream.
    """
    def __init__(self, window_size=15, step_size=3, ema_alpha=0.1):
        """
        Parameters:
        - window_size: 15 sampel (~3.0 detik pada 5Hz)
        - step_size: 3 sampel (~0.6 detik per slide)
        - ema_alpha: Koefisien filter Exponential Moving Average untuk tracking gravitasi
        """
        self.window_size = window_size
        self.step_size = step_size
        self.ema_alpha = ema_alpha

    def _decompose_signals_causal(self, df_trip):
        """
        Menghitung sinyal terdekomposisi vertikal/horizontal secara kausal (tanpa masa depan).
        """
        df = df_trip.copy()
        
        ax = df["accelerometerX"].values if "accelerometerX" in df.columns else np.zeros(len(df))
        ay = df["accelerometerY"].values if "accelerometerY" in df.columns else np.zeros(len(df))
        az = df["accelerometerZ"].values if "accelerometerZ" in df.columns else np.zeros(len(df))
        
        gx = df["gyroX"].values if "gyroX" in df.columns else np.zeros(len(df))
        gy = df["gyroY"].values if "gyroY" in df.columns else np.zeros(len(df))
        gz = df["gyroZ"].values if "gyroZ" in df.columns else np.zeros(len(df))
        
        speed = df["speed"].values if "speed" in df.columns else np.zeros(len(df))
        
        n = len(df)
        if n == 0:
            return pd.DataFrame()
            
        # 1. Causal Gravity Estimation via Exponential Moving Average (EMA)
        g_vec = np.zeros((n, 3))
        g_curr = np.array([ax[0], ay[0], az[0]])
        
        for i in range(n):
            a_curr = np.array([ax[i], ay[i], az[i]])
            g_curr = (1.0 - self.ema_alpha) * g_curr + self.ema_alpha * a_curr
            g_vec[i] = g_curr
            
        g_norm = np.linalg.norm(g_vec, axis=1, keepdims=True) + 1e-6
        g_unit = g_vec / g_norm
        
        acc_3d = np.column_stack([ax, ay, az])
        
        # 2. Dynamic Vertical Shock (Sejajar arah gravitasi)
        a_vert = np.sum(acc_3d * g_unit, axis=1)
        a_vert_dyn = np.abs(a_vert - g_norm.squeeze())
        
        # 3. Dynamic Horizontal Forces (Tegak lurus gravitasi: belok, rem, akselerasi)
        a_horiz_dyn = np.linalg.norm(acc_3d - a_vert[:, None] * g_unit, axis=1)
        
        # 4. Total Magnitudes
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # 5. Jerk (Turunan perubahan percepatan vertikal & angular)
        jerk_vert = np.abs(np.diff(a_vert_dyn, prepend=a_vert_dyn[0]))
        gyro_jerk = np.abs(np.diff(gyro_mag, prepend=gyro_mag[0]))
        
        # Simpan ke DataFrame hasil
        df["vert_dyn"] = a_vert_dyn
        df["horiz_dyn"] = a_horiz_dyn
        df["acc_mag"] = acc_mag
        df["gyro_mag"] = gyro_mag
        df["jerk_vert"] = jerk_vert
        df["gyro_jerk"] = gyro_jerk
        df["speed_clean"] = np.clip(speed, 0.0, None)
        
        return df

    def _extract_window_features(self, window_df):
        """
        Mengekstrak 47 fitur statistik & sinyal domain dari 1 window data (v2).
        """
        features = {}
        
        signals = [
            "vert_dyn",      # Dynamic Vertical Shock (Karakteristik utama lubang)
            "horiz_dyn",     # Dynamic Lateral/Longitudinal Forces (Belokan/Pengereman)
            "acc_mag",       # Total Accelerometer Magnitude
            "gyro_mag",      # Total Gyroscope Magnitude
            "jerk_vert",     # Vertical Jerk (Hentakan tajam)
            "gyro_jerk",     # Gyroscope Jerk (Hentakan oleng roda)
            "speed_clean"    # Kecepatan Kendaraan
        ]
        
        for sig in signals:
            vals = window_df[sig].values
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            rng_val = float(np.ptp(vals))
            rms_val = np.sqrt(np.mean(vals**2))
            
            features[f"{sig}_mean"] = float(mean_val)
            features[f"{sig}_std"] = float(std_val)
            features[f"{sig}_min"] = float(min_val)
            features[f"{sig}_max"] = float(max_val)
            features[f"{sig}_range"] = float(rng_val)
            features[f"{sig}_rms"] = float(rms_val)
            
        # Fitur Rasio & Sinyal Lanjutan v2
        max_v = features["vert_dyn_max"]
        rms_v = features["vert_dyn_rms"]
        max_h = features["horiz_dyn_max"]
        mean_spd = features["speed_clean_mean"]
        
        # Crest Factor Vertikal: Peak to RMS ratio
        features["crest_factor_vert"] = float(max_v / (rms_v + 1e-6))
        
        # Rasio Vertikal terhadap Horizontal (Pothole dominan di vertikal)
        features["vert_to_horiz_ratio"] = float((max_v + 1e-6) / (max_h + 1e-6))
        
        # Speed-Normalized Shock: Normalisasi kekuatan goncangan terhadap laju kendaraan
        features["speed_norm_vert_shock"] = float(max_v / (mean_spd + 1.5))
        
        # Zero-Crossing Rate pada sinyal vertikal
        v_vals = window_df["vert_dyn"].values
        centered_v = v_vals - np.median(v_vals)
        zcr = ((centered_v[:-1] * centered_v[1:]) < 0).sum() / (len(v_vals) - 1 + 1e-6)
        features["vert_zcr"] = float(zcr)
        
        # Indikator Kendaraan Bergerak vs Berhenti
        features["is_moving"] = float(1.0 if mean_spd >= 1.5 else 0.0)
        
        return features

    def extract_from_trips(self, trips_list):
        """
        Mengekstrak fitur dari seluruh perjalanan dengan pipeline V2.
        """
        records = []
        labels = []
        groups = []
        meta_list = []
        
        for trip_df in trips_list:
            if len(trip_df) < self.window_size:
                continue
                
            trip_id = trip_df["trip_id"].iloc[0] if "trip_id" in trip_df.columns else "unknown"
            df_processed = self._decompose_signals_causal(trip_df)
            
            n_samples = len(df_processed)
            for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
                end_idx = start_idx + self.window_size
                win = df_processed.iloc[start_idx:end_idx]
                
                feat = self._extract_window_features(win)
                records.append(feat)
                
                # Window dilabeli 1 jika terdapat kejadian lubang di dalam window
                win_label = 1 if (win["label"] == 1).any() else 0
                labels.append(win_label)
                groups.append(trip_id)
                
                meta_list.append({
                    "trip_id": trip_id,
                    "timestamp": win["timestamp"].iloc[-1] if "timestamp" in win.columns else 0,
                    "latitude": win["latitude"].mean() if "latitude" in win.columns else 0.0,
                    "longitude": win["longitude"].mean() if "longitude" in win.columns else 0.0,
                    "peak_shock": feat["vert_dyn_max"],
                    "speed": feat["speed_clean_mean"],
                })
                
        X = pd.DataFrame(records)
        y = pd.Series(labels, name="label")
        groups = pd.Series(groups, name="trip_id")
        metadata = pd.DataFrame(meta_list)
        
        return X, y, groups, metadata

    def extract_from_stream(self, df_stream):
        """
        Mengekstrak fitur dari data sensor stream / database (Real-time Inference V2).
        """
        if len(df_stream) < self.window_size:
            return pd.DataFrame(), pd.DataFrame()
            
        df_processed = self._decompose_signals_causal(df_stream)
        records = []
        meta_list = []
        
        n_samples = len(df_processed)
        for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            win = df_processed.iloc[start_idx:end_idx]
            
            feat = self._extract_window_features(win)
            records.append(feat)
            
            meta_list.append({
                "timestamp": win["timestamp"].iloc[-1] if "timestamp" in win.columns else 0,
                "latitude": win["latitude"].mean() if "latitude" in win.columns else 0.0,
                "longitude": win["longitude"].mean() if "longitude" in win.columns else 0.0,
                "peak_shock": feat["vert_dyn_max"],
                "speed": feat["speed_clean_mean"],
            })
            
        X = pd.DataFrame(records)
        metadata = pd.DataFrame(meta_list)
        return X, metadata
