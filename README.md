# Autonomous Vehicle Simulator — Diffusion Policy

Simulator interaktif berbasis [PyBullet](https://pybullet.org/) yang mendemonstrasikan **Diffusion Policy** untuk navigasi kendaraan otonom. Kendaraan membuat keputusan berdasarkan observasi lingkungan secara real-time, dengan pipeline lengkap ditampilkan di terminal setiap frame.

---

## Instalasi

```bash
git clone https://github.com/arfandiq/uasavkel6_2026.git
cd uasavkel6_2026
pip install -r requirements.txt
```

---

## Jalankan

```bash
python main.py
```

| Key | Fungsi |
|-----|--------|
| `C` | Toggle action chunking |
| `Q` | Quit |

---

## Pipeline Architecture

Kendaraan menggunakan pipeline 5 stage untuk membuat keputusan kontrol:

```
Observation → Encoder → Conditional Diffusion Model → Action Sequence → Vehicle Control
```

### Stage 1: Observation

Ekstrak raw state kendaraan menjadi vector 8 dimensi:

```
[pos_x, pos_y, vel_x, vel_y, yaw, speed, heading_error, dist_to_target]
```

`heading_error` dihitung sebagai sudut antara heading saat ini dan arah ke target waypoint. Observasi adalah **input utama** yang menentukan seluruh keputusan kontrol.

### Stage 2: Observation Encoder

Mengkonversi 8D raw observation menjadi 4D compact features:

| Feature | Formula | Range |
|---------|---------|-------|
| `heading_error_norm` | `clip(heading_error / π)` | [-1, 1] |
| `dist_norm` | `clip(dist / 5.0)` | [0, 1] |
| `speed_norm` | `clip(speed / MAX_SPEED)` | [0, 1] |
| `lateral_vel_norm` | `clip(vel_y / MAX_SPEED)` | [-1, 1] |

Dimensionality reduction 8D → 4D memastikan policy fokus pada fitur paling relevan.

### Stage 3: Conditional Diffusion Model

Action **digenrasikan langsung** dari encoded observations melalui proses diffusion:

1. **Action generation** — Steering proportional terhadap heading error, throttle berdasarkan jarak dan kecepatan
2. **Iterative denoising** — 10 iterasi refinement menggunakan Stochastic Langevin Dynamics:

```
refined = current_action + α × (target - current_action) + noise
```

Dimana `α = 0.05` dan `noise ~ N(0, 0.02)`. Setiap iterasi memperbaiki aksi menuju distribusi optimal dengan stochasticity untuk menjelajahi multimodal distribution.

### Stage 4: Action Sequence

Output berupa urutan aksi (steering, throttle, brake). Dengan **action chunking** (`ACTION_CHUNK_SIZE = 8`), model memprediksi 8 langkah aksi sekaligus dan di-averaging untuk trajectory yang lebih smooth.

### Stage 5: Vehicle Control

Aksi diterapkan dengan **receding horizon control** — setelah aksi diterapkan, pipeline berjalan ulang pada frame berikutnya dengan observasi baru, menciptakan closed-loop feedback.

Position Control menggunakan smooth ramp acceleration (0.3 m/s per frame) yang meniru proses iterative refinement.

---

## Terminal Output

Setiap frame menampilkan pipeline stages dengan warna:

```
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
[OBSERVATION] pos=(1.23,2.45) vel=(2.30,1.50) yaw=0.45 heading_err=0.15 dist_to_target=2.50
[ENCODER] normalized=[0.05,0.50,0.18,0.10] → 4D features extracted
[CONDITIONAL_DIFFUSION] iter 10/10 denoising: action_seq=[(0.79,0.04),...]  (Stochastic Langevin Dynamics)
[ACTION_SEQUENCE] predicted=[steering:0.79, throttle:0.04, brake:0.00]
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
[AUTONOMOUS] Heading to WP3/7 | Distance: 1.45m | Speed: 12.30 m/s
```

| Label | Warna | Stage |
|-------|-------|-------|
| `[OBSERVATION]` | Cyan | Raw state extraction |
| `[ENCODER]` | Yellow | Feature extraction |
| `[CONDITIONAL_DIFFUSION]` | Blue | Action generation + denoising |
| `[ACTION_SEQUENCE]` | Red | Predicted actions |
| `[VEHICLE]` | Red | Control signals |
| `[MODE1-SMOOTH]` | Green | Smooth acceleration |
| `[AUTONOMOUS]` | Blue | Waypoint progress |

---

## Konfigurasi

Parameter utama di `av_config.py`:

```python
MAX_SPEED = 20.0          # m/s — kecepatan maksimum
MAX_STEER = 0.8           # rad (~46°) — sudut steering maksimum
ACTION_CHUNK_SIZE = 8     # Jumlah langkah aksi per chunk
WAYPOINT_THRESHOLD = 0.3  # m — jarak waypoint dianggap tercapai
SIMULATION_STEP = 1/240.0 # 240 Hz physics timestep
NUM_WAYPOINTS = 8         # Jumlah waypoints dalam pola lingkaran
ARENA_SIZE = 10.0         # m — ukuran area simulasi
```

---

## Project Structure

```
uasav_kel6/
├── main.py                    # Entry point
├── av_config.py               # Konfigurasi global
├── av_environment.py          # PyBullet scene (plane, grid, waypoints)
├── av_vehicle.py              # Racecar model + physics
├── av_controller.py           # Pure Pursuit waypoint controller
├── av_input_simple.py         # Keyboard input handler
├── av_simulator.py            # Main simulator orchestrator
├── av_diffusion_pipeline.py   # Diffusion Policy pipeline (core)
├── requirements.txt           # Dependencies
└── README.md                  # Dokumentasi
```

---

## Troubleshooting

**Kendaraan tidak bergerak**
- Pastikan jendela PyBullet dalam fokus (klik window)
- Cek `MAX_SPEED > 0` di `av_config.py`

**Kendaraan wheelie / nungging**
- Kurangi throttle di `av_diffusion_pipeline.py`
- Pastikan `vz = 0` di `apply_action()`

**Waypoint tidak tercapai**
- Cek `WAYPOINT_THRESHOLD` di `av_config.py`
- Pastikan `current_wp_idx` di-reset ke 0 saat startup

**Terminal output tidak muncul**
- Pipeline output ditampilkan setiap frame dalam mode autonomous

---

## Technology

- **Physics Engine:** [PyBullet](https://pybullet.org/) — real-time rigid body simulation
- **Vehicle Model:** Racecar URDF dari pybullet_data
- **Concepts:** Stochastic Langevin Dynamics, Action Chunking, Receding Horizon Control

---

## Referensi

- Chi, C. et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* arXiv:2303.04137v5. [Paper](https://diffusion-policy.cs.columbia.edu)

---

**Catatan:** Ini adalah demonstrasi edukasi yang mengadaptasi konsep-konsep Diffusion Policy menggunakan heuristic rules, bukan neural network penuh. Tujuannya adalah memvisualisasikan pipeline dan konsep secara interaktif untuk pembelajaran.
