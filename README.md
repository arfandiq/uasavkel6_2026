# Simulator Kendaraan Otonom — Demo Diffusion Policy

Simulator berbasis PyBullet yang mendemonstrasikan konsep-konsep kunci dari paper **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"** (Chi et al., 2023).

## Ringkasan

Simulator ini mengimplementasikan tiga demo interaktif yang menampilkan konsep pembelajaran mesin dari paper Diffusion Policy. Demo-demo ini menghubungkan teori visuomotor policy learning dengan simulasi robotika praktis, ideal untuk tujuan pendidikan dan memahami tantangan pembelajaran policy di dunia nyata.

**Referensi Paper:** arXiv:2303.04137v5  
**Penulis:** Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song

## Mulai Cepat

### Instalasi

```bash
# Clone atau download repository ini
cd uasav_kel6

# Opsi 1: Menggunakan pip (Python standar)
pip install -r requirements.txt

# Opsi 2: Menggunakan Anaconda
conda create -n av-simulator python=3.10
conda activate av-simulator
pip install -r requirements.txt
```

### Jalankan Simulator

```bash
python main.py
```

Setelah jendela PyBullet terbuka, klik untuk fokus, lalu tekan tombol untuk mengontrol kendaraan.

---

## Konsep Paper & Pemetaan Demo

### Teori Inti dari Paper Diffusion Policy

Paper Diffusion Policy memperkenalkan pendekatan baru terhadap pembelajaran visuomotor policy menggunakan **Denoising Diffusion Probabilistic Models (DDPMs)**. Alih-alih langsung memprediksi aksi robot, policy mempelajari gradient field dari distribusi aksi dan iteratif menyaring aksi noisy melalui **Stochastic Langevin Dynamics**.

**Keunggulan utama Diffusion Policy:**
- **Distribusi aksi multimodal** — menangani banyak solusi valid untuk satu observasi
- **Ruang aksi berdimensi tinggi** — prediksi full sequence, bukan single action
- **Pelatihan stabil** — tidak memerlukan negative sampling seperti policy berbasis energi
- **Kontrol receding horizon** — replanning closed-loop untuk eksekusi robust

---

## Tiga Demo Dijelaskan

### Demo 1: Perbandingan Mode (Position vs Velocity Control)

**Konsep Paper:** Stochastic Langevin Dynamics & Optimasi Berbasis Gradient

Dalam Diffusion Policy, policy mempelajari melakukan iterative gradient descent pada distribusi aksi: `x' = x - γ∇E(x)` (Persamaan 2 di paper).

**Apa yang ditampilkan:**
- **Mode 1 (Position Control):** Ramp akselerasi smooth (0.3 m/s per frame)
- **Mode 2 (Velocity Control):** Perubahan kecepatan instant/jerky

Setiap frame di Mode 1 seperti satu iterasi gradient descent di mana ramp akselerasi adalah learning rate dan target speed adalah energy minimum. Mode 2 melewati langkah gradient → perilaku jerky, tidak stabil (suboptimal menurut paper).

**Kontrol:**
```
1 → Position Control (smooth, stabil)
2 → Velocity Control (instant, jerky)
W → Uji gerakan maju
```

**Output Terminal:**
```
[MODE1-SMOOTH] ramp speed → 14.66 m/s (target: 14.67)
[MODE2-INSTANT] jump to 14.66 m/s (instant!)
```

---

### Demo 2: Behavior Cloning & Replay

**Konsep Paper:** Imitation Learning & Policy Learning dari Demonstrasi

Paper Diffusion Policy dibangun atas imitation learning: "Policy learning dari demonstrasi, dalam bentuk paling sederhana, dapat diformulasikan sebagai tugas supervised regression untuk memetakan observasi ke aksi."

**Apa yang ditampilkan:**
- Rekam trajectory driving manual (pasangan observasi + aksi)
- Replay trajectory yang dipelajari secara otonom
- Robot belajar meniru demonstrasi manusia

Keunggulan paper: Diffusion model dapat mengekspresikan distribusi multimodal. Jika ada banyak cara valid melakukan task, model mempelajari semuanya.

**Kontrol:**
```
SPACE → Toggle MANUAL dan AUTONOMOUS modes
R → Rekam demonstrasi
P → Replay demonstrasi yang direkam
WASD → Drive manual (saat merekam)
```

**Output Terminal:**
```
[INFO] MULAI MEREKAM demonstrasi (Behavior Cloning)...
[INFO] Demonstrasi selesai! 150 steps tersimpan.
[INFO] REPLAY demonstrasi (150 steps)
```

---

### Demo 3: Autonomous Navigation (Receding Horizon Control)

**Konsep Paper:** Receding Horizon Control & Action Chunking

Paper menyatakan: "Closed-loop action sequences. Kami menggabungkan kemampuan policy memprediksi action sequence berdimensi tinggi dengan receding-horizon control untuk mencapai eksekusi robust... continuously re-plan aksinya dalam manner closed-loop sambil mempertahankan temporal action consistency."

**Apa yang ditampilkan:**
- Kendaraan navigasi otonom 8 waypoint dalam pola circular
- Akselerasi smooth (Position Control) untuk trajectory konsisten
- Replanning real-time dengan pure pursuit controller
- Visualisasi action chunk 8-step (sphere oranye)

**Detail implementasi:**
- `ACTION_CHUNK_SIZE = 8` → prediksi 8 langkah ahead (visible sebagai sphere oranye)
- Setiap frame: replan berdasarkan vehicle state saat ini (closed-loop)
- Position control ramp mempertahankan temporal consistency
- Real-time state feedback, bukan trajectory following open-loop

**Output Terminal:**
```
[VEHICLE] throttle=0.98, steering=-0.05, speed=14.66, steer_angle=-0.04
[MODE1-SMOOTH] ramp speed → 14.66 m/s (target: 14.66)
[AUTONOMOUS] Heading to WP0/7 | Distance: 2.15m | Speed: 14.50 m/s
[AUTONOMOUS] Heading to WP1/7 | Distance: 1.80m | Speed: 14.48 m/s
...
[AUTONOMOUS] ALL WAYPOINTS COMPLETED!
```

---

## Pemetaan Konsep Paper ke Implementasi

| Konsep Paper | Matematis | Demo 1 | Demo 2 | Demo 3 |
|---|---|---|---|---|
| **Stochastic Langevin Dynamics** | x' = x - γ∇E(x) | ✓ Smooth ramp | - | ✓ Position control |
| **Multimodal Distributions** | Learn complex p(a\|o) | - | ✓ Multiple styles | - |
| **Imitation Learning** | Behavior cloning | - | ✓ Record & replay | ✓ Pure pursuit |
| **Action Chunking** | Predict N future actions | - | - | ✓ 8-step lookahead |
| **Receding Horizon** | Replan every step | - | - | ✓ Continuous replanning |
| **Training Stability** | No negative sampling | ✓ Smooth vs jerky | - | ✓ Consistent trajectory |

---

## Struktur Project

```
uasav_kel6/
├── main.py                 ← Entry point (jalankan ini)
├── av_config.py           ← Konfigurasi global
├── av_environment.py      ← Scene PyBullet (plane, grid, waypoints, goal)
├── av_vehicle.py          ← Model Racecar (physics, control modes)
├── av_controller.py       ← Pure pursuit controller (waypoint navigation)
├── av_input_simple.py     ← Keyboard + Xbox input handler
├── av_simulator.py        ← Main simulator orchestrator (modes, recording, replay)
├── requirements.txt       ← Dependensi Python
└── README.md              ← File ini
```

---

## Konfigurasi Kunci

Edit `av_config.py` untuk customize:

```python
MAX_SPEED = 15.0              # m/s — kecepatan maksimal kendaraan
MAX_STEER = 0.8               # rad (~46°) — batas steering angle
ACTION_CHUNK_SIZE = 8         # N-step prediction (demo 3)
WAYPOINT_THRESHOLD = 0.3      # m — jarak untuk consider waypoint "tercapai"
NUM_WAYPOINTS = 8             # Pola navigasi circular
ARENA_SIZE = 10.0             # m — ukuran grid
```

---

## Kontrol

### Keyboard

```
WASD / Arrow Keys     Gerakkan mobil maju/mundur/belok
SPACE                 Toggle MANUAL ↔ AUTONOMOUS mode
R                     Rekam demonstrasi (demo 2)
P                     Replay demonstrasi (demo 2)
C                     Toggle action chunking (demo 3)
1                     Position Control mode (smooth, demo 1)
2                     Velocity Control mode (jerky, demo 1)
Q                     Quit
```

### Xbox Controller (jika tersedia)

```
Left Stick X          Steering
Triggers (L/R)        Throttle (RT forward, LT backward)
D-Pad                 Input alternatif
```

---

## Penggunaan yang Dimaksudkan

### Untuk Pendidikan
Pahami bagaimana kontrol berbasis gradient berbeda dari perintah kecepatan langsung, lihat behavior cloning dalam aksi, dan visualisasikan receding horizon control dengan action chunking.

### Untuk Penelitian Robotika
Foundation untuk menguji implementasi diffusion policy, baseline untuk membandingkan strategi kontrol, referensi untuk integrasi robot real-world.

### Untuk Implementasi Diffusion Policy
Simulator ini menyediakan:
1. Perbandingan Position vs Velocity Control (analisis stabilitas dari paper Section 4.2)
2. Pipeline Behavior Cloning (rekam demonstrasi → learn policy → replay)
3. Receding Horizon + Action Chunking (framework planning closed-loop)

Implementasi real akan:
- Ganti pure pursuit dengan neural network diffusion policy
- Pelajari dari demonstrasi yang direkam (saat ini hanya replay)
- Gunakan observasi visual (camera images) dengan visual conditioning
- Prediksi action sequence dan refine via denoising iterations

---

## Troubleshooting

### Kendaraan tidak bergerak
- Cek `MAX_SPEED > 0` di `av_config.py`
- Pastikan jendela PyBullet memiliki focus
- Verifikasi input keyboard (coba semua WASD/arrow keys)

### Kendaraan wheelies/tipping
- Kurangi `MAX_STEER` atau `MAX_SPEED`
- Tingkatkan position control smoothing (sudah di 0.3 m/s per frame)

### Sphere oranye tidak terlihat
- Ini adalah trajectory preview spheres di demo 3
- Hanya muncul di AUTONOMOUS mode dengan action chunking enabled
- Coba tekan `C` untuk toggle chunking visibility

### Terminal output terlalu banyak
- Output intentional untuk observe control signals per-frame
- Membantu pahami gradient descent (ramp speed) real-time

---

## Catatan Performa

- **Simulasi 240 Hz** — cocok dengan robot control loop real-world
- **Capable real-time** — berjalan ~1x speed di hardware modern
- **GPU acceleration** — optional via pybullet (lihat `av_environment.py`)

---

## Referensi

- **Paper:** [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu)
- **Penulis:** Chi et al. (2023) — arXiv:2303.04137v5
- **Code:** Available di diffusion-policy.cs.columbia.edu
- **PyBullet:** [Bullet Physics Simulation](https://pybullet.org/)

---

