# Perancangan Arsitektur Sistem & Infrastruktur Deployment Model HyperGrid

Dokumen ini menyusun rekomendasi arsitektur sistem, infrastruktur, biaya komputasi, strategi error handling, serta metode klustering dan autentikasi untuk penerapan model Machine Learning **HyperGrid** (Pothole & Road Condition Detection) berdasarkan rincian teknis dari model `hg-model` dan referensi `paper.pdf`.

---

## 1. Strategi Pipeline Optimal (Edge Mobile vs Cloud Server)

Sistem HyperGrid mengadopsi pendekatan **Hybrid Edge-Cloud Architecture (Edge-Heavy Sensing, Cloud-Heavy Aggregation)**.

```
┌─────────────────────────────────────────────────────────┐
│              EDGE (Mobile Application)                  │
│  - Raw Sensor Collection (Acc, Gyro, GPS @ 5Hz/10Hz)    │
│  - Sliding Window & Causal Feature Extraction (EMA)     │
│  - Real-time Inference (ONNX Runtime Engine)            │
│  - Local Deduplication & Anomaly Buffering              │
└────────────────────────────┬────────────────────────────┘
                             │ Batch Sync (JSON/Protobuf over HTTPS/MQTT)
                             ▼
┌─────────────────────────────────────────────────────────┐
│             CLOUD (Server & Database)                   │
│  - Authentication & Ingestion Gateway                   │
│  - Geospatial Spatial Clustering (DBSCAN / HDBSCAN)     │
│  - Road Quality Indexing & Heatmap Generation           │
│  - Model Retraining & OTA Update Server                 │
└─────────────────────────────────────────────────────────┘
```

### Pembagian Tugas Pipeline:
1. **Edge (Mobile App - React Native / Native Mobile):**
   - **Tugas:** Menangkap data inersial ($A_x, A_y, A_z, G_x, G_y, G_z$) dan GPS, melakukan ekstraksi fitur (*rolling median / EMA causal gravity tracking*, *vertical-horizontal decomposition*, *jerk*, statistik window), serta mengeksekusi inferensi ML menggunakan **ONNX Runtime Mobile**.
   - **Alasan:** Ukuran model ONNX HyperGrid sangat kecil (**~41.1 KB** untuk V1 dengan 32 fitur, dan **~60 KB** untuk V2 dengan 47 fitur). Inferensi di Edge menghemat konsumsi bandwidth hingga 99% (mengirim data agregat lubang/anomali saja, bukan raw sensor 10Hz) serta tidak membutuhkan koneksi internet *real-time* saat kendaraan melaju di area *blank spot*.

2. **Cloud Server (Backend Infrastructure):**
   - **Tugas:** Ingest data deteksi yang dikirim mobile, validasi autentikasi, klustering spasial (agregasi laporan lubang dari banyak pengguna di koordinat berdekatan), pemetaan *heatmap* jalan, serta retrain model secara berkala.

---

## 2. Arsitektur Server dan Topologi (Event-Driven Bus & Microservices)

Untuk menangani aliran data (*high-throughput ingestion*) dari ribuan perangkat kendaraan bergerak, server dirancang menggunakan topologi **Event-Driven Architecture (EDA)**.

```
                               ┌─────────────────┐
                               │   Mobile App    │
                               └────────┬────────┘
                                        │ HTTPS / WSS / MQTT
                                        ▼
                               ┌─────────────────┐
                               │ API Gateway     │
                               │ (Kong / Nginx)  │
                               └────────┬────────┘
                                        │ Auth Check & Rate Limit
                                        ▼
                               ┌─────────────────┐
                               │ Ingestion Svc   │
                               └────────┬────────┘
                                        │ Publish Events
                                        ▼
                      ┌───────────────────────────────────┐
                      │ Event Bus / Message Broker        │
                      │ (Apache Kafka / Redpanda / NATS)  │
                      └─┬───────────────────────────────┬─┘
                        │ Topic: road.detections        │ Topic: road.raw_telemetry
                        ▼                               ▼
      ┌───────────────────────────────────┐   ┌───────────────────────────┐
      │ Spatial Clustering Service        │   │ Cold Storage Worker       │
      │ (Go / Python Worker + HDBSCAN)    │   │ (MinIO / AWS S3)          │
      └─────────────────┬─────────────────┘   └───────────────────────────┘
                        │ Write Clustered Potholes
                        ▼
      ┌───────────────────────────────────┐
      │ Database Cluster                  │
      │ (PostgreSQL + PostGIS)            │
      └─────────────────┬─────────────────┘
                        │ Read Analytics / Heatmap
                        ▼
      ┌───────────────────────────────────┐
      │ Dashboard API / Client Portal     │
      └─────────────────┬─────────────────┘
```

### Komponen Utama Topologi:
- **API Gateway & Auth Service:** Menangani enkripsi TLS, *rate limiting*, serta validasi JWT token dari aplikasi mobile.
- **Message Broker / Event Bus (Apache Kafka / NATS / RabbitMQ):** Menampung event `RoadEventDetected` secara asinkron tanpa memblokir koneksi HTTP mobile.
- **Spatial Clustering Worker:** Service independen yang mengonsumsi stream event dari Kafka, menghitung kedekatan lokasi lubang, dan memperbarui status kondisi jalan di database secara *batch*.
- **PostgreSQL + PostGIS:** Database relational spatial untuk menyimpan koordinat lubang, segmen jalan, dan *user profiles*.

---

## 3. Estimasi Cost Computation (Edge Mobile & Cloud Server)

### A. Resource Consumption pada Mobile Device (Edge)
- **Model Size:** ~41.1 KB (V1) – ~60 KB (V2).
- **RAM Usage:** **< 15 MB** tambahan (RAM runtime ONNX Engine & Sensor Buffer).
- **CPU Usage:** **< 2% - 4%** dari 1 core CPU (Inferensi dijalankan dengan stride 3-5 sampel, yaitu ~2-3 kali per detik).
- **Battery Consumption:** **~1.5% - 3% per jam** (Mayoritas dikonsumsi oleh chip GPS, bukan komputasi ONNX ML).
- **Network Bandwidth:** **< 50 KB / trip 10 km** (Hanya mengirim titik anomali yang lolos *threshold* `0.47`, bukan data sensor mentah).

### B. Computational Cost & Specs pada Server (10.000 Active Drivers)
- **Ingestion & Cluster Workers:** 2 Node Instance (e.g., AWS `c6g.large` - 2 vCPU, 4GB RAM) $\approx$ **\$30 - \$50 / bulan**.
- **Event Bus (NATS / Kafka Managed):** 1 Instance (2 vCPU, 4GB RAM) $\approx$ **\$25 / bulan**.
- **PostgreSQL + PostGIS Database:** AWS RDS PostgreSQL `db.m6g.large` (2 vCPU, 8GB RAM, 100GB Storage) $\approx$ **\$80 - \$120 / bulan**.
- **Total Estimasi Server:** **\$135 - \$195 / bulan** untuk melayani 10.000 pengguna aktif.

---

## 4. Pemilihan Database Strategy

Untuk mendukung kebutuhan kueri spasial (lokasi GPS jalan) dan *time-series telemetry*, kombinasi database berikut direkomendasikan:

| Peran Database | Teknologi Rekomendasi | Rationale & Kegunaan |
| :--- | :--- | :--- |
| **Primary Relational & Spatial DB** | **PostgreSQL + PostGIS** | **Wajib.** Mendukung fungsi geospasial native seperti `ST_DWithin`, `ST_ClusterDBSCAN`, dan `ST_MakeLine` untuk agregasi titik lubang pada segmen jalan. |
| **BaaS / Rapid Prototyping** | **Supabase** (PostgreSQL-backed) | Sesuai dengan konfigurasi `pull_data.py` pada repo `hg-model`. Memudahkan autentikasi, Row Level Security (RLS), dan Realtime Subscriptions. |
| **TimeSeries / Telemetry (Optional)** | **TimescaleDB Extension** / **ClickHouse** | Digunakan jika sistem memperluas fitur untuk menyimpan seluruh rekaman histori guncangan jalan (*raw sensor telemetry*) untuk audit kualitas infrastruktur. |
| **Cache & Real-time State** | **Redis** | Menyimpan sesi login user, *rate limiting*, dan *buffer caching* sementara untuk geofencing aktif. |

---

## 5. Metode dan Cara Clustering (Spatial Deduplication Pothole)

Pada paper dan eksperimen `hg-model`, deteksi dari satu kendaraan saat melewati lubang dapat memicu 3-5 window berturut-turut. Selain itu, 100 kendaraan yang melewati lubang yang sama akan menghasilkan 100+ titik koordinat yang sedikit bergeser karena akurasi GPS HP (margin error 3–10 meter).

### Metode Clustering: **DBSCAN / HDBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

```
             Titik Deteksi GPS Mentah              Hasil Clustering DBSCAN
             
                  •    •                                ┌─────────┐
                •   •   •    (Deteksi User A)           │ Cluster │ ──► 1 Lubang Valid
                  •    •     (Deteksi User B)           │    #1   │     (Centroid + Dynamic
               •    •   •    (Deteksi User C)           └─────────┘      Confidence Tier)
                                                             
             • (Noise / Single False Positive) ─────────► Discard / Unverified
```

### Cara Kerja Clustering:
1. **Spatiotemporal Windowing:**
   - Ambil data deteksi baru dengan *bounding box* geografis (Radius $Eps = 5-10\text{ meter}$).
2. **Cluster Formation (DBSCAN):**
   - **Core Point Threshold ($MinPts \ge 3$):** Sebuah titik lubang dikonfirmasi sebagai **"Lubang Valid"** jika dilaporkan oleh minimal 3 kejadian deteksi berbeda (atau multiple trips/users) dalam radius 8 meter.
3. **Point Centroid & Confidence Tier Weighting:**
   - Menghitung koordinat centroid $\bar{L} = (\text{Mean Lat}, \text{Mean Long})$ dari kluster.
   - Menghitung bobot keparahan (*Peak Shock* $a_v$ dan *Confidence Tier* dari model ONNX) untuk menentukan tingkat urgensi perbaikan jalan (*Low*, *Medium*, *High*, *Critical*).
4. **PostGIS Query Example:**
   ```sql
   SELECT ST_ClusterDBSCAN(geom, eps => 0.00008, minpoints => 3) OVER () AS cluster_id,
          AVG(latitude) AS lat, AVG(longitude) AS lng, MAX(peak_shock) AS max_shock
   FROM pothole_detections
   WHERE created_at >= NOW() - INTERVAL '7 days';
   ```

---

## 6. Arsitektur Autentikasi dan Keamanan (Auth & Security Strategy)

Sistem menangani data spasial sensitif (rute perjalanan pengguna) dan pengiriman telemetry.

```
┌───────────────┐                  ┌────────────────┐                  ┌─────────────────┐
│ Mobile App    │ ──1. Login ────► │ Auth Provider  │ ──2. Token ────► │ Client App      │
│ (React Native)│                  │ (OAuth2/JWT)   │   (Access+Refresh)│ (Secure Store)  │
└───────┬───────┘                  └────────────────┘                  └────────┬────────┘
        │                                                                       │
        └──────────────3. Send Pothole Payload + Bearer JWT ────────────────────┘
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │ API Gateway / Ingestion│
                           │  - Verify Signature    │
                           │  - Extract Device ID   │
                           │  - Anonymize GPS Track │
                           └────────────────────────┘
```

### Strategi Keamanan & Auth:
1. **Authentication Protocol:**
   - **OAuth2.0 / OIDC with JWT (JSON Web Tokens):** Pengguna login via Supabase Auth / Auth0. Access Token memiliki masa berlaku pendek (15-60 menit) dengan Refresh Token terenkripsi di *Secure Store* HP (Keychain / Keystore).
2. **Anonymous & Privacy-Preserving Telemetry (Geospatial Anonymization):**
   - **Trips Stripping:** Koordinat GPS 200 meter awal dan akhir perjalanan (area rumah/kantor pengguna) secara otomatis dipotong (*truncated*) pada sisi mobile sebelum dikirim ke server untuk menjaga privasi lokasi asal/tujuan.
   - **Payload Ingestion Token:** Data deteksi jalan hanya menyertakan `anonymous_device_id` dan data guncangan, tanpa menempelkan identitas pribadi pengguna pada histori rute.
3. **Data Integrity & Anti-Spoofing:**
   - Payload dikirim dengan **HMAC-SHA256 Signature** yang digenerate oleh app mobile untuk mencegah *injection* koordinat lubang palsu oleh *bot*.

---

## 7. Strategi Penanganan Error & Resiliensi (Fault Tolerance & Error Recovery)

Sistem dirancang dengan resiliensi tinggi (*graceful degradation*) pada 3 layer utama:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FAULT TOLERANCE STRATEGY                           │
├──────────────────────┬──────────────────────────┬───────────────────────┤
│    MOBILE SENSOR     │    NETWORK / OFFLINE     │     SERVER BACKEND    │
├──────────────────────┼──────────────────────────┼───────────────────────┤
│ • Sensor Fallback    │ • SQLite Offline Buffer  │ • Dead Letter Queue   │
│ • Noise Filtering    │ • Exponential Backoff    │ • Circuit Breaker     │
│ • Graceful Degradation│ • Batch Flush on WiFi   │ • Database Fallback   │
└──────────────────────┴──────────────────────────┴───────────────────────┘
```

### A. Error Sisi Mobile & Sensor (Edge Level)
1. **Sensor Disconnection / Hardware Failure:**
   - Jika `accelerometer` atau `gyroscope` berhenti memberikan data, `RoadDetectionService` menangkap exception, menghentikan loop inferensi, dan mencoba *re-subscribe* sensor hingga 3x. Jika tetap gagal, berikan notifikasi subtle ke pengguna ("Sensor inersial tidak merespon").
2. **GPS Signal Loss (Tunnel / Urban Canyon):**
   - Jika GPS kehilangan sinyal (`latitude=0.0, longitude=0.0`), simpan sampel deteksi di memori sementara (RAM) dan gunakan koordinat GPS terakhir yang diketahui (*Last Known Location*) atau lakukan *dead reckoning* berdasarkan akumulasi kecepatan hingga sinyal GPS kembali *lock*.

### B. Error Sisi Jaringan & Offline Mode (Connectivity Level)
1. **No Internet Connection (Offline Buffering):**
   - Jika koneksi internet terputus saat lubang terdeteksi, payload **TIDAK dibuang**. Payload disimpan ke database lokal HP (**SQLite / WatermelonDB / AsyncStorage**).
2. **Sync Retry & Exponential Backoff:**
   - Saat koneksi terhubung kembali, aplikasi mengunggah data tertunda secara *batch* menggunakan algoritma *Exponential Backoff with Jitter* untuk mencegah *thundering herd effect* pada server.

### C. Error Sisi Server & Data Processing (Backend Level)
1. **Message Broker Down / Ingestion Failure:**
   - Menggunakan pattern **Dead Letter Queue (DLQ)** pada Kafka/RabbitMQ. Event yang gagal diproses oleh *clustering worker* dipindahkan ke DLQ untuk dievaluasi ulang tanpa menghentikan antrean event utama.
2. **Database Connection Pool Exhaustion:**
   - Implementasi **Circuit Breaker Pattern** (e.g. menggunakan Resilience4j atau Envoy Proxy). Jika PostgreSQL mengalami lonjakan beban, API Gateway memberikan respon `202 Accepted` dan menyimpan data mentah ke *cold storage* (Object Storage S3) untuk di-ingest ulang (*backfill*) saat database kembali normal.
3. **Database Fallback di Repo (`case_current_database.py`):**
   - Seperti yang sudah diimplementasikan pada kode repositori, jika koneksi Supabase gagal/timeout, sistem secara otomatis beralih ke *fallback local dataset* (`/data/Pothole/trip1_sensors.csv`) sehingga sistem analisis/simulasi tetap berjalan tanpa crashing.

---

## Kesimpulan Ringkasan Arsitektur

| Komponen | Pilihan Arsitektur / Teknologi |
| :--- | :--- |
| **Pipeline Strategy** | Hybrid (Inference @ Edge Mobile ONNX, Aggregation & Cluster @ Server Cloud) |
| **Model Runtime Mobile** | ONNX Runtime React Native / Web (~41KB-60KB, RAM < 15MB) |
| **Server Topology** | Event-Driven Architecture (API Gateway $\rightarrow$ NATS/Kafka $\rightarrow$ Workers $\rightarrow$ DB) |
| **Database** | PostgreSQL + PostGIS (Secondary: Redis & Supabase BaaS) |
| **Clustering Method** | Spatial Density-Based Clustering (DBSCAN / HDBSCAN dengan radius 5-10m) |
| **Auth & Privacy** | OAuth2 JWT + HMAC Payload Signature + GPS Trip Truncation |
| **Error Handling** | Offline SQLite Buffering, Circuit Breaker, Dead Letter Queue (DLQ), Local Fallback |
