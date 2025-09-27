# import cv2
# import os
# import time
# import pygame
# from detectors.people_detector import PeopleDetector
# from utils.cloudinary_uploader import upload_frame
# from utils.twilio_notifier import send_whatsapp_message

# class CrowdMonitor:
#     def __init__(self, threshold=10, alert_sound="sounds/alert.mp3"):
#         self.detector = PeopleDetector()
#         self.threshold = threshold

#         # Initialize pygame mixer for sound alert
#         pygame.mixer.init()
#         self.alert_sound = pygame.mixer.Sound(alert_sound)

#         # Alert cooldown to avoid spam
#         self.last_alert_time = 0
#         self.alert_cooldown = 5  # seconds

#     def process_stream(self, video_source=0):
#         cap = cv2.VideoCapture(video_source)

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Detect people + check clusters
#             count, boxes, cluster_alert = self.detector.detect_people(frame)

#             # Draw bounding boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box)
#                 color = (0, 0, 255) if cluster_alert else (0, 255, 0)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             # Show info text
#             cv2.putText(frame, f"Count: {count}", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#             cv2.putText(frame, f"Cluster Alert: {cluster_alert}", (20, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if cluster_alert else (0, 255, 0), 2)

#             # Display feed
#             cv2.imshow("Crowd Monitor", frame)

#             # If overcrowd detected in a cluster
#             if cluster_alert:
#                 current_time = time.time()
#                 if current_time - self.last_alert_time > self.alert_cooldown:
#                     print(f"🚨 Cluster Overcrowding detected! ({count} people total)")

#                     # 🔊 Play alert sound
#                     self.alert_sound.play()

#                     # Save frame temporarily
#                     frame_path = "overcrowd.jpg"
#                     cv2.imwrite(frame_path, frame)

#                     # Upload & send alert
#                     url = upload_frame(frame_path)
#                     if url:
#                         msg_text = f"🚨 Overcrowding Alert!\nDetected {count} people, with cluster size >= {self.threshold}."
#                         send_whatsapp_message(url, msg_text)

#                     # Remove temp file
#                     if os.path.exists(frame_path):
#                         os.remove(frame_path)

#                     # Update cooldown
#                     self.last_alert_time = current_time

#             # Quit with 'q'
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

import cv2
import os
import time
import pygame
from detectors.people_detector import PeopleDetector
from utils.cloudinary_uploader import upload_frame
from utils.twilio_notifier import send_whatsapp_message

class CrowdMonitor:
    def __init__(self, threshold=10, alert_sound="sounds/alert.mp3"):
        self.detector = PeopleDetector()
        self.threshold = threshold

        # Initialize pygame mixer for sound alert
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound(alert_sound)

        # Alert cooldown to avoid spam
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds

    def process_stream(self, video_source=0):
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect people + clusters
            count, boxes, clusters, labels, congestion_detected = self.detector.detect_people(
                frame,
                eps=100,
                min_samples=5,
                min_cluster_size=self.threshold
            )

            # Draw bounding boxes for each person
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)  # green
                if labels is not None and labels[i] != -1:
                    cluster_id = labels[i]
                    if len(clusters.get(cluster_id, [])) >= self.threshold:
                        color = (0, 0, 255)  # red for congested cluster
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw bounding rectangles around congested clusters
            for cluster_id, members in clusters.items():
                if len(members) >= self.threshold:
                    xs = [boxes[i][0] for i in members]
                    ys = [boxes[i][1] for i in members]
                    xe = [boxes[i][2] for i in members]
                    ye = [boxes[i][3] for i in members]
                    cv2.rectangle(frame,
                                  (int(min(xs)), int(min(ys))),
                                  (int(max(xe)), int(max(ye))),
                                  (0, 0, 255), 3)
                    cv2.putText(frame,
                                f"Cluster {cluster_id}: {len(members)}",
                                (int(min(xs)), int(min(ys)) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            # Info text
            cv2.putText(frame, f"Count: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Congestion: {congestion_detected}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if congestion_detected else (0, 255, 0), 2)

            # Display video feed
            cv2.imshow("Crowd Monitor", frame)

            # 🚨 Trigger Alert
            if congestion_detected:
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    print(f"🚨 Overcrowding detected! ({count} people total)")

                    # 🔊 Play alert sound
                    self.alert_sound.play()

                    # Save frame temporarily
                    frame_path = "overcrowd.jpg"
                    cv2.imwrite(frame_path, frame)

                    # Upload to Cloudinary
                    url = upload_frame(frame_path, "Stampede")
                    
                    if url:
                        msg_text = f"⚠ Overcrowding Alert!\nDetected {count} people, with cluster size >= {self.threshold}."
                        send_whatsapp_message(url, msg_text)

                    # Clean up temp file
                    if os.path.exists(frame_path):
                        os.remove(frame_path)

                    # Update cooldown
                    self.last_alert_time = current_time

            # Quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
