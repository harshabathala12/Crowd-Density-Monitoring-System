# Crowd Density Monitoring System (CDMS)
### Autonomous Real-Time Situational Awareness System

![Status](https://img.shields.io/badge/System-Online-green)
![Tech](https://img.shields.io/badge/AI-YOLOv8-blue)
![Hardware](https://img.shields.io/badge/GPU-Accelerated-7020ff)
![License](https://img.shields.io/badge/Engineering-Final_Year_Project-orange)


## 🎬 Demo Video

[![CDMS Demo](https://img.shields.io/badge/Watch-CDMS_Demo-blue?style=for-the-badge)](https://github.com/harshabathala12/Crowd-Density-Monitoring-System/releases/download/v1.0/CDMS_Demo1.mp4)


---

## 📌 Project Overview

**CDMS** is an edge-optimized engineering solution designed to prevent crowd crush events in real time.  
Unlike passive surveillance recording, this system utilizes a **Deterministic Logic Engine (v2.0)** and **Computer Vision** to proactively identify dangerous congestion levels and trigger automated intervention protocols.

Designed for high-density environments such as **streets, stadiums, and transit hubs**, the system prioritizes **low latency** and **fault tolerance** over theoretical complexity. It is engineered to run on **local edge hardware** with GPU acceleration.

---

## 🚀 Key Engineering Features

1. **GPU-Accelerated Perception**
   - Utilizes **NVIDIA CUDA** cores for YOLOv8 inference
   - Ensures seamless processing on consumer hardware

2. **Lag-Optimized Pipeline**
   - Implements a **Frame-Skipping Algorithm** (Process 1 / Skip 2)
   - Reduces CPU load while maintaining stable tracking continuity

3. **Deterministic Logic Engine**
   - Replaces black-box AI predictions with transparent, physics-based heuristics
   - Combines **Crowd Density** and **Optical Flow Flux** to significantly reduce false positives

4. **Active Alert System**
   - **Anti-Flapping Cooldown Mechanism (60s)** to prevent alert fatigue
   - Automated **Email alerts via SMTP** for critical hazard events

5. **Operational Directives**
   - Dashboard provides **context-aware commands**
   - Example: *“Halt Inflow”* instead of generic warnings

---

## 📐 System Architecture
*Optimized real-time pipeline (GPU-accelerated)*

```text
Input Stream
    ↓
Frame Pre-Processor (Resize / Skip)
    ↓
Perception Engine (YOLOv8 GPU) ──→ Motion Engine (Optical Flow)
    ↓                              ↓
    ╰────── Deterministic Logic Engine v2.0 ──────╯
                     ↓
            Decision Matrix (Thresholds)
            ↙          ↓          ↘
      UI Dashboard   SMTP Server   Visualizer

```
---

## 🛠️ Tech Stack

- **Core:** Python 3.10
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Backend:** Flask (WebSockets / Streaming)
- **Hardware Support:** NVIDIA CUDA (PyTorch)
- **Frontend:** HTML5, TailwindCSS (Cyberpunk / Ops Theme)

---

## 📂 Project Structure

```text
CDMS_Final/
│
├── application/
│   ├── templates/
│   │   └── index.html          # Operational Dashboard UI
│   ├── analytics_engine.py    # Deterministic Logic v2.0
│   └── app.py                 # Main Application Entry Point
│
├── data/
│   └── raw_videos/
│       ├── sample.mp4         # Normal Condition Test
│       ├── sample1.mp4        # Critical Condition Test
│
├── config.py                  # Configuration Settings
├── requirements.txt           # Dependency List
├── yolov8n.pt                 # YOLOv8 Nano Model
└── README.md                  # Documentation

```
---
## ⚙️ Installation & Setup

### 1️⃣ Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended)

### 2️⃣ Install Dependencies
⚠️ Critical: Install GPU-enabled PyTorch first.

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows
.\venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Install other dependencies
pip install -r requirements.txt
```
### 3️⃣ Setup
Ensure the yolov8n.pt model file is present in the root directory.

---

## 🖥️ Usage

### 1️⃣ Start the System
Run the main application:
```bash
python application/app.py
```
### 2️⃣ Access Dashboard
Open your browser at: `http://127.0.0.1:5000`

### 3️⃣ Test Scenarios
- Modify `VIDEO_SOURCE` in `app.py`:
   - `sample.mp4` → Stable crowd flow (Green / Normal)
   - `sample1.mp4` → Critical density spike & alert triggers

---

## 📊 System Thresholds

| Status   | Index Range | UI Feedback       | Action Triggered              |
|----------|------------|-----------------|-------------------------------|
| NORMAL   | 0 – 29     | Green / Neon    | Passive Monitoring            |
| WARNING  | 30 – 49    | Orange          | Deploy Stewards               |
| CRITICAL | 50 – 100   | Red / Blinking  | Email Alert + Halt Traffic    |

---

## ✅ Outcomes

- Demonstrates that Deterministic heuristics combined with computer vision provide robust performance in simulated safety-critical scenarios.
- Real-time operation with GPU acceleration for near-instantaneous alerts
- Stable performance during simulated street-crossing hazards.

--- 

## ⚠️ Limitations

- Occlusion: Extreme crowd overlap may reduce absolute person count (partially mitigated via optical flow).
- Camera Angle: Optimized for elevated CCTV views (~45° downward).
- Lighting: Accuracy drops in low-light environments without IR support.

--- 

## 🛡️ Credits

- Developed as a multi-year engineering research and development project
- Methodology: Deterministic Heuristics & Computer Vision
- Objective: Autonomous Crowd Safety

---

