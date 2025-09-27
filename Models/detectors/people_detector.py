import torch
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

class PeopleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Using YOLOv8 on {self.device}")
        self.model.to(self.device)

    def detect_people(self, frame, eps=80, min_samples=5, min_cluster_size=8):
        results = self.model.predict(frame, device=self.device, verbose=False)

        boxes, centers = [], []
        for r in results[0].boxes:
            if int(r.cls) == 0:  # Person
                xyxy = r.xyxy.cpu().numpy().tolist()[0]
                boxes.append(xyxy)
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                centers.append([cx, cy])

        clusters, congestion_detected, labels = {}, False, None
        if len(centers) >= min_samples:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(centers)
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                members = np.where(labels == cluster_id)[0]
                clusters[cluster_id] = members
                if len(members) >= min_cluster_size:
                    congestion_detected = True

        return len(boxes), boxes, clusters, labels, congestion_detected
