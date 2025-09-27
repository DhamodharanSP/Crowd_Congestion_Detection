import cv2
import time
import pygame
from detectors.people_detector import PeopleDetector

class CrowdMonitor:
    def __init__(self, threshold=8, alert_sound="sounds/alert.mp3"):
        self.detector = PeopleDetector()
        self.threshold = threshold
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound(alert_sound)
        self.last_alert_time = 0
        self.alert_cooldown = 5

    def process_frame(self, frame):
        count, boxes, clusters, labels, congestion_detected = self.detector.detect_people(
            frame,
            eps=80,
            min_samples=5,
            min_cluster_size=self.threshold
        )
        # Trigger alert if congested
        if congestion_detected and (time.time() - self.last_alert_time > self.alert_cooldown):
            print(f"[ALERT] Congestion detected! ({count} people)")
            self.alert_sound.play()
            self.last_alert_time = time.time()
        return count, boxes, clusters, labels, congestion_detected
