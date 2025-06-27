from ultralytics import YOLO
from boxmot import StrongSort
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import torch
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cosine

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.embedding_memory = {}  # fixed_id -> dict with embedding, bbox, last_seen
        self.max_ids_allowed = 22
        self.next_fixed_id = 1

        self.model = YOLO(model_path)
        reid_weights_path = Path("trackers/weights/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150.pth")

        self.tracker = StrongSort(
            reid_weights=reid_weights_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half=False,
            max_cos_dist=0.25,
            max_iou_dist=0.4,
            max_age=120,
            n_init=1,
            nn_budget=200,
            ema_alpha=0.95,
            mc_lambda=0.98
        )

        print("✅ StrongSort with ReID initialized.")

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _find_similar_id(self, new_emb, new_bbox, current_frame):
        best_match = None
        best_score = float('inf')

        for fixed_id, data in self.embedding_memory.items():
            emb = data['embedding']
            old_bbox = data['bbox']
            last_seen = data['last_seen']

            time_decay = max(1.0, (current_frame - last_seen) / 100.0)
            if time_decay > 8:
                continue

            emb_dist = cosine(new_emb, emb)
            foot_dist = np.linalg.norm(
                np.array(get_foot_position(new_bbox)) - np.array(get_foot_position(old_bbox))
            )

            score = emb_dist + 0.01 * foot_dist + 0.05 * time_decay

            if score < best_score and emb_dist < 0.85 and foot_dist < 250:
                best_score = score
                best_match = fixed_id

        return best_match

    def _assign_fixed_id(self, embedding, bbox, current_frame):
        if isinstance(embedding, (int, float)):
            raise TypeError("❌ Embedding must be a vector, not a track ID or scalar.")

        match_id = self._find_similar_id(embedding, bbox, current_frame)
        if match_id is not None:
            self.embedding_memory[match_id]["embedding"] = embedding
            self.embedding_memory[match_id]["bbox"] = bbox
            self.embedding_memory[match_id]["last_seen"] = current_frame
            return match_id

        if len(self.embedding_memory) >= self.max_ids_allowed:
            match_id = self._find_similar_id(embedding, bbox, current_frame)
            if match_id is not None:
                self.embedding_memory[match_id]["embedding"] = embedding
                self.embedding_memory[match_id]["bbox"] = bbox
                self.embedding_memory[match_id]["last_seen"] = current_frame
                return match_id
            else:
                # Force reuse of oldest ID if no match
                oldest_id = min(self.embedding_memory.items(), key=lambda item: item[1]["last_seen"])[0]
                print(f"[FORCED REUSE] Max IDs reached. Reassigning oldest ID {oldest_id}")
                self.embedding_memory[oldest_id] = {
                    "embedding": embedding,
                    "bbox": bbox,
                    "last_seen": current_frame
                }
                return oldest_id

        fixed_id = self.next_fixed_id
        self.next_fixed_id += 1
        self.embedding_memory[fixed_id] = {
            "embedding": embedding,
            "bbox": bbox,
            "last_seen": current_frame
        }
        print(f"[NEW FIXED ID] Assigned Fixed ID {fixed_id}")
        return fixed_id

    def cleanup_old_ids(self, current_frame, max_age=900):
        self.embedding_memory = {
            fid: data for fid, data in self.embedding_memory.items()
            if current_frame - data["last_seen"] <= max_age
        }




    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 10
        detections = []
        for i in range(0, len(frames), batch_size):
            resized_batch = [cv2.resize(f, (1280, 720)) for f in frames[i:i + batch_size]]
            detections_batch = self.model.predict(resized_batch, conf=0.3, iou=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detections_ultra = detection.boxes.xyxy.cpu().numpy()
            confidences = detection.boxes.conf.cpu().numpy()
            class_ids = detection.boxes.cls.cpu().numpy().astype(int)

            detection_input = []
            for i in range(len(detections_ultra)):
                x1, y1, x2, y2 = detections_ultra[i]
                conf = confidences[i]
                cls_id = class_ids[i]
                cls_name = cls_names[cls_id]
                if cls_name == "goalkeeper":
                    cls_id = cls_names_inv["player"]
                detection_input.append([x1, y1, x2, y2, conf, cls_id])

            detection_input = np.array(detection_input)
            track_results = self.tracker.update(detection_input, frame=frames[frame_num])

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for track in track_results:
                x1, y1, x2, y2, track_id, cls_id = track
                bbox = [x1, y1, x2, y2]

                if cls_id == cls_names_inv["player"]:
                    embedding = self.tracker.get_last_embedding()
                    fixed_id = self._assign_fixed_id(embedding, bbox, frame_num)


                    if fixed_id is None:
                        continue
                    tracks["players"][frame_num][fixed_id] = {"bbox": bbox, "team_color": (255, 0, 0) if fixed_id <= 11 else (0, 0, 255)}

                elif cls_id == cls_names_inv["referee"]:
                    continue

                elif cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    

    



    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)),
                    angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y], [x - 10, y - 20], [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = (team_ball_control_till_frame == 1).sum()
        team_2_num_frames = (team_ball_control_till_frame == 2).sum()
        total = team_1_num_frames + team_2_num_frames

        if total == 0:
            team_1, team_2 = 0.0, 0.0
        else:
            team_1 = team_1_num_frames / total
            team_2 = team_2_num_frames / total

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames

    def save_id_consistency_plot(self, tracks, save_path="output_videos/id_consistency_plot.png"):
        id_to_frames = {}
        for frame_num, players in enumerate(tracks["players"]):
            for track_id in players:
                if track_id not in id_to_frames:
                    id_to_frames[track_id] = []
                id_to_frames[track_id].append(frame_num)

        plt.figure(figsize=(12, 6))
        for track_id, frames in id_to_frames.items():
            plt.plot(frames, [track_id]*len(frames))

        plt.xlabel("Frame Number")
        plt.ylabel("Track ID")
        plt.title("Player ID Consistency Over Time")
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Saved ID consistency visualization at {save_path}")
