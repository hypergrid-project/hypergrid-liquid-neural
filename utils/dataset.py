import os
import glob
import pandas as pd
import numpy as np

# Path absolut ke root direktori proyek
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_data_dir(data_dir):
    """Membantu resolusi direktori data agar bekerja dari CWD mana pun."""
    if not os.path.isabs(data_dir):
        if not os.path.exists(data_dir):
            alt_path = os.path.join(PROJECT_ROOT, data_dir.lstrip("./"))
            if os.path.exists(alt_path):
                return alt_path
            else:
                return os.path.join(PROJECT_ROOT, "data")
    return data_dir


def clean_trip_sensor_data(df_sensor, trip_name="trip"):
    """
    Membersihkan dan memvalidasi data sensor mentah (Pipeline V2).
    
    Langkah Pembersihan:
    1. Pengurutan sekuensial timestamp & eliminasi timestamp duplikat.
    2. Perbaikan anomali kecepatan (speed < 0 -> interpolasi & clip).
    3. Imputasi nilai hilang (NaN/Inf) pada kanal sensor.
    4. Pengecekan clipping batas fisik sensor (outlier noise).
    """
    df = df_sensor.copy()
    
    # 1. Pastikan terurut dan tanpa duplikasi timestamp
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    # 2. Pembersihan Kecepatan (GPS Speed)
    speed_col = "speed" if "speed" in df.columns else ("speed_kmh" if "speed_kmh" in df.columns else None)
    if speed_col:
        # Kecepatan negatif (misal -1.0 pada GPS no-fix) diubah menjadi NaN lalu diinterpolasi
        neg_mask = df[speed_col] < 0
        if neg_mask.any():
            df.loc[neg_mask, speed_col] = np.nan
            df[speed_col] = df[speed_col].interpolate(method="linear").bfill().ffill()
        df[speed_col] = df[speed_col].clip(lower=0.0, upper=200.0)
        df["speed"] = df[speed_col]
    else:
        df["speed"] = 0.0

    # 3. Kanal Akselerometer & Giroskop
    sensor_cols = {
        "accelerometerX": ["sensor_x", "ax"],
        "accelerometerY": ["sensor_y", "ay"],
        "accelerometerZ": ["sensor_z", "az"],
        "gyroX": ["sensor_gx", "gx"],
        "gyroY": ["sensor_gy", "gy"],
        "gyroZ": ["sensor_gz", "gz"],
    }
    
    for std_name, alt_names in sensor_cols.items():
        if std_name not in df.columns:
            found = False
            for alt in alt_names:
                if alt in df.columns:
                    df[std_name] = df[alt]
                    found = True
                    break
            if not found:
                df[std_name] = 0.0
                
        # Tangani NaN/Inf & outlier ekstrem (> 80 m/s^2 atau ~8G)
        df[std_name] = df[std_name].replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill()
        if "accelerometer" in std_name:
            df[std_name] = df[std_name].clip(lower=-80.0, upper=80.0)
        elif "gyro" in std_name:
            df[std_name] = df[std_name].clip(lower=-30.0, upper=30.0)
            
    return df


def load_raw_trips(data_dir="./data", pothole_tolerance_sec=0.75):
    """
    Memuat data mentah dari folder Pothole dan RoadCondition secara terisolasi per file/trip (Pipeline V1).
    """
    data_dir = _resolve_data_dir(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan di '{data_dir}'.")

    trips = []
    
    # 1. Muat trip dari folder Pothole
    pothole_dir = os.path.join(data_dir, "Pothole")
    sensor_files = sorted(glob.glob(os.path.join(pothole_dir, "*_sensors.csv")))
    
    for sf in sensor_files:
        trip_name = os.path.basename(sf).replace("_sensors.csv", "")
        pf = os.path.join(pothole_dir, f"{trip_name}_potholes.csv")
        
        df_sensor = pd.read_csv(sf)
        df_sensor = df_sensor.sort_values(by="timestamp").reset_index(drop=True)
        df_sensor["trip_id"] = f"pothole_{trip_name}"
        df_sensor["label"] = 0
        
        if os.path.exists(pf):
            df_potholes = pd.read_csv(pf)
            if "timestamp" in df_potholes.columns and not df_potholes.empty:
                pothole_ts = df_potholes["timestamp"].values
                for ts in pothole_ts:
                    mask = (df_sensor["timestamp"] >= (ts - pothole_tolerance_sec)) & \
                           (df_sensor["timestamp"] <= (ts + pothole_tolerance_sec))
                    df_sensor.loc[mask, "label"] = 1
        
        trips.append(df_sensor)
        
    # 2. Muat trip dari folder RoadCondition
    road_dir = os.path.join(data_dir, "RoadCondition")
    good_files = sorted(glob.glob(os.path.join(road_dir, "good*_sensors.csv")))
    for gf in good_files:
        trip_name = os.path.basename(gf).replace("_sensors.csv", "")
        df_good = pd.read_csv(gf)
        df_good = df_good.sort_values(by="timestamp").reset_index(drop=True)
        df_good["trip_id"] = f"road_{trip_name}"
        df_good["label"] = 0
        trips.append(df_good)
        
    bad_files = sorted(glob.glob(os.path.join(road_dir, "bad*_sensors.csv")))
    for bf in bad_files:
        trip_name = os.path.basename(bf).replace("_sensors.csv", "")
        df_bad = pd.read_csv(bf)
        df_bad = df_bad.sort_values(by="timestamp").reset_index(drop=True)
        df_bad["trip_id"] = f"road_{trip_name}"
        df_bad["label"] = 0
        trips.append(df_bad)
        
    if not trips:
        raise ValueError(f"Tidak ada file CSV sensor (*_sensors.csv) yang ditemukan di {data_dir}.")

    return trips


def load_raw_trips_v2(data_dir="./data", pothole_tolerance_sec=0.6, verbose=True):
    """
    Memuat dan membersihkan data perjalanan dengan pipeline V2 yang diperkuat (Data Cleaning & Quality Control).
    
    Parameters:
    - data_dir: Path direktori data
    - pothole_tolerance_sec: Toleransi waktu lubang (default: 0.6s untuk sinkronisasi tajam pada 5Hz)
    - verbose: Apakah menampilkan log detail per trip
    
    Returns:
    - List of DataFrames (cleaned trips)
    """
    data_dir = _resolve_data_dir(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan di '{data_dir}'.")

    trips = []
    total_cleaned_anomalies = 0
    
    # 1. Folder Pothole
    pothole_dir = os.path.join(data_dir, "Pothole")
    sensor_files = sorted(glob.glob(os.path.join(pothole_dir, "*_sensors.csv")))
    
    for sf in sensor_files:
        trip_name = os.path.basename(sf).replace("_sensors.csv", "")
        pf = os.path.join(pothole_dir, f"{trip_name}_potholes.csv")
        
        df_raw = pd.read_csv(sf)
        df_clean = clean_trip_sensor_data(df_raw, trip_name=trip_name)
        df_clean["trip_id"] = f"pothole_{trip_name}"
        df_clean["label"] = 0
        df_clean["is_pothole_event"] = 0
        
        if os.path.exists(pf):
            df_potholes = pd.read_csv(pf)
            if "timestamp" in df_potholes.columns and not df_potholes.empty:
                pothole_ts = df_potholes["timestamp"].values
                for ts in pothole_ts:
                    # Toleransi terpusat untuk ground truth
                    mask = (df_clean["timestamp"] >= (ts - pothole_tolerance_sec)) & \
                           (df_clean["timestamp"] <= (ts + pothole_tolerance_sec))
                    df_clean.loc[mask, "label"] = 1
                    
                    # Titik waktu terdekat ditandai sebagai event center
                    nearest_idx = (df_clean["timestamp"] - ts).abs().idxmin()
                    df_clean.loc[nearest_idx, "is_pothole_event"] = 1
                    
        trips.append(df_clean)
        
    # 2. Folder RoadCondition (Good Roads)
    road_dir = os.path.join(data_dir, "RoadCondition")
    good_files = sorted(glob.glob(os.path.join(road_dir, "good*_sensors.csv")))
    for gf in good_files:
        trip_name = os.path.basename(gf).replace("_sensors.csv", "")
        df_clean = clean_trip_sensor_data(pd.read_csv(gf), trip_name=trip_name)
        df_clean["trip_id"] = f"road_{trip_name}"
        df_clean["label"] = 0
        df_clean["is_pothole_event"] = 0
        trips.append(df_clean)
        
    # 3. Folder RoadCondition (Bad Roads)
    bad_files = sorted(glob.glob(os.path.join(road_dir, "bad*_sensors.csv")))
    for bf in bad_files:
        trip_name = os.path.basename(bf).replace("_sensors.csv", "")
        df_clean = clean_trip_sensor_data(pd.read_csv(bf), trip_name=trip_name)
        df_clean["trip_id"] = f"road_{trip_name}"
        df_clean["label"] = 0
        df_clean["is_pothole_event"] = 0
        trips.append(df_clean)
        
    if not trips:
        raise ValueError(f"Tidak ada file sensor CSV yang valid di {data_dir}.")

    if verbose:
        print(f"Total trips V2 berhasil dimuat & dibersihkan: {len(trips)} trips")
        pothole_trips_count = sum(1 for t in trips if (t["label"] == 1).any())
        print(f" -> Trips berisi pothole: {pothole_trips_count} trips")
        print(f" -> Trips non-pothole (Good/Bad roads): {len(trips) - pothole_trips_count} trips")

    return trips
