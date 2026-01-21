# 🤖 Robot Movement Monitor

Track your robot’s movement in real time using ArUco markers and an Intel RealSense camera. The system classifies motion as **MOVING**, **STATIONARY**, or **MARKER LOST**, and overlays velocity, FPS, marker ID, and trail directly on the video feed.

---

## ✨ Features
- 🧭 Real‑time ArUco marker detection and tracking
- 🟢 State classification: MOVING / STATIONARY / MARKER LOST
- 📈 Velocity estimation with smoothing
- 🖼️ On‑screen overlay with FPS, ID, coordinates, trail
- 🧾 Optional logging to file
- 🧩 Built‑in marker generator (single or sheet)

---

## 🧰 Requirements
- Python 3.8+
- Intel RealSense camera (D435 or similar)
- OpenCV with ArUco support

# 🤖 Robot Movement Monitor

Track your robot’s movement in real time using ArUco markers and an Intel RealSense camera. The system classifies motion as **MOVING**, **STATIONARY**, or **MARKER LOST**, and overlays velocity, FPS, marker ID, and trail directly on the video feed.

---

## ✨ Features
- 🧭 Real‑time ArUco marker detection and tracking
- 🟢 State classification: MOVING / STATIONARY / MARKER LOST
- 📈 Velocity estimation with smoothing
- 🖼️ On‑screen overlay with FPS, ID, coordinates, trail
- 🧾 Optional logging to file
- 🧩 Built‑in marker generator (single or sheet)

---

## 🧰 Requirements
- Python 3.8+
- Intel RealSense camera (D435 or similar)
- OpenCV with ArUco support

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** RealSense support requires `pyrealsense2`. Install it separately if needed:
>
> ```bash
> pip install pyrealsense2
> ```

---

## 🏁 Quick Start
### 1) Generate a marker

```bash
python generate_marker.py --id 0 --size 300
```

Print and attach the marker to your robot.

### 2) Run the monitor

```bash
python robot_monitor.py
```

Controls:
- `q` → Quit
- `r` → Reset tracking history
- `s` → Save current frame

---

## ⚙️ Configuration
Edit [config.py](config.py) to tune:
- camera resolution and FPS
- marker dictionary
- movement thresholds
- overlay settings

> ✅ **Important:** `generate_marker.py` uses `DICT_6X6_250` by default. Ensure `MARKER_DICT` in [config.py](config.py) matches the marker dictionary you generate.

---

## 📁 Project Structure
```
.
├── robot_monitor.py      # Main tracking application
├── generate_marker.py    # Marker generator
├── config.py             # Settings and thresholds
├── requirements.txt      # Dependencies
└── markers/              # Generated markers
```

---

## 📸 Tips for Best Tracking
- Use good lighting 💡
- Keep the marker flat and clearly visible 👀
- Use a 5–10 cm printed marker for 1–3 meters distance

---

## 📜 License
MIT (feel free to use and modify)

---

## 🙌 Credits
Built with OpenCV ArUco and Intel RealSense.
