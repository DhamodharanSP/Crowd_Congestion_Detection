import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

class PeopleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using Device: {self.device}")
        self.model.to(self.device)

    def detect_people(self, frame, eps=100, min_samples=10, min_cluster_size=10):
        """
        Detect people and check for clusters.
        Returns:
            count (int): total people detected
            boxes (list): bounding boxes
            clusters (dict): cluster_id -> member indices
            labels (ndarray): DBSCAN labels per detected person
            congestion_detected (bool): True if cluster >= min_cluster_size
        """
        results = self.model(frame, device=self.device, verbose=False)

        boxes, centers = [], []
        for r in results[0].boxes:
            if int(r.cls) == 0:  # class 0 = person
                xyxy = r.xyxy.cpu().numpy().tolist()[0]
                boxes.append(xyxy)
                # center point
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                centers.append([cx, cy])

        clusters, congestion_detected, labels = {}, False, None
        if len(centers) >= min_samples:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
            labels = clustering.labels_
            for cluster_id in set(labels):
                if cluster_id == -1:  # noise
                    continue
                members = np.where(labels == cluster_id)[0]
                clusters[cluster_id] = members
                if len(members) >= min_cluster_size:
                    congestion_detected = True

        return len(boxes), boxes, clusters, labels, congestion_detected
    