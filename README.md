# ⚽ Soccer Player Re-Identification and Tracking using YOLOv11 and StrongSORT

This repository presents a computer vision pipeline for **soccer player tracking and ID assignment** across frames using object detection and ReID-based tracking. It uses **YOLOv11** for detection and **StrongSORT** for tracking with ReID embeddings. While the system aims to maintain consistent identities, occasional ID switches may still occur.

---

## ⚙️ Setup Instructions

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/YA-shiKa/player-reid-tracking.git
cd player-reid-tracking
python -m venv venv
venv\Scripts\activate     # For Windows
# OR
source venv/bin/activate  # For Mac/Linux
pip install -r requirements.txt
```

### 2. Download Required Files

- 🔍 **YOLOv11 Detection Model**: `yolov11_players.pt`
  - ➤ Place it in the `models/` folder

- 🧠 **ReID Model for StrongSORT**: `osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150.pth`
  - ➤ Place it inside `trackers/weights/`

- 📹 **Input Video**: `match_clip.mp4`
  - ➤ Put it inside `input_videos/`

---

## ▶️ Running the Code

```bash
python main.py
```

- ✅ Output video will be saved at: `output_videos/output_video.avi`
- 📊 ID consistency plot: `output_videos/id_consistency_plot.png`

---

## 🎥 Demo Video



https://github.com/user-attachments/assets/2a2983b5-bcbe-4d64-a79a-3f687c3dcf49



## 📦 Dependencies
Installable via:

```bash
pip install -r requirements.txt
```

Key Libraries:
- `ultralytics==8.x` (for YOLOv11)
- `boxmot` (StrongSORT tracking)
- `opencv-python`
- `torch`, `numpy`, `pandas`
- `matplotlib`, `scikit-learn`

---

## 🧠 Report Summary

### ✅ Approach and Methodology

- Used **YOLOv11** to detect players, ball, and referees per frame.
- Used **StrongSORT** with ReID for multi-object tracking.
- Briefly experimented with **MARS embedding model** for ReID, but it was removed due to inferior performance.
- Implemented a **Fixed ID Memory** system:
  - Stores ReID embeddings, foot positions, and last-seen timestamps
  - Reassigns consistent IDs using cosine similarity, foot distance, and time-decay scoring

### 🧪 Techniques Tried

| Technique                         | Purpose                             | Outcome                                         |
|----------------------------------|-------------------------------------|-------------------------------------------------|
| StrongSORT + ReID                | Multi-object tracking               | Worked well for real-time detection             |
| Cosine similarity + foot distance| Re-identify players accurately      | Effective but ID switches still occurred        |
| Time-decay matching              | Prevent stale re-matching           | Reduced stale IDs                               |
| Max-ID limit + reuse             | Limit total IDs to 22               | Controlled inflation, but led to reused IDs     |
| ID consistency plot              | Visual debug of IDs over time       | Helped reveal inconsistencies                   |
| MARS embedding (removed)         | Alternate ReID strategy             | Tried but less consistent than StrongSORT       |


### 🚧 Challenges Encountered

- **Embedding inconsistency**: ReID vectors fluctuated due to motion blur or occlusion.
- **ID switching**: Some players received new IDs upon re-entry.
- **Small scale**: Players far from the camera were difficult to track.
- **Limited training**: No football-specific ReID training.
- **Forced ID reuse**: Max-ID cap sometimes overwrote active players.

### 🧩 Incomplete? What's Left

The system performs re-identification but ID consistency is imperfect. In some cases, it assigns >22 unique IDs despite constraints.

**Future improvements:**
- Fine-tune ReID model on football-specific datasets
- Add motion prediction (Kalman filters, LSTM, etc.)
- Leverage attention-based temporal modeling
- Implement real-time ID correction logic

---

### 🎓 Author

**Yashika Maligi**  

---
