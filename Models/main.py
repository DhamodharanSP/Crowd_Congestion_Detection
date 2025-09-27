import cv2
from face_recognition_module.FaceRecognition import load_known_faces, recognize_faces
from utils.video_utils import blur_region, draw_box
from modules.CrowdMonitor import CrowdMonitor

def process_camera_streams(camera_sources, known_embeddings, known_names):
    caps = [cv2.VideoCapture(src) for src in camera_sources]
    crowd_monitor = CrowdMonitor(threshold=8)

    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret: continue

            # --- Face recognition + blur unknown ---
            recognized_faces = recognize_faces(frame, known_embeddings, known_names)
            for name, score, box in recognized_faces:
                if box is not None:
                    if name == "Unknown":
                        frame = blur_region(frame, box)
                    else:
                        frame = draw_box(frame, box, name, score)

            # --- Congestion detection ---
            count, boxes, clusters, labels, congestion = crowd_monitor.process_frame(frame)

            # Draw congestion clusters
            for cluster_id, members in clusters.items():
                if len(members) >= crowd_monitor.threshold:
                    xs = [boxes[i][0] for i in members]
                    ys = [boxes[i][1] for i in members]
                    xe = [boxes[i][2] for i in members]
                    ye = [boxes[i][3] for i in members]
                    cv2.rectangle(frame, (int(min(xs)), int(min(ys))),
                                  (int(max(xe)), int(max(ye))), (0,0,255), 3)
                    cv2.putText(frame, f"Cluster {cluster_id}: {len(members)}",
                                (int(min(xs)), int(min(ys))-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Info overlay
            cv2.putText(frame, f"People: {count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
            cv2.putText(frame, f"Congestion: {congestion}", (20,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255) if congestion else (0,255,0),2)

            cv2.imshow(f"Camera {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


def main():
    known_folder = "known_faces"
    known_embeddings, known_names = load_known_faces(known_folder)
    camera_sources = [0]  # add more cameras if needed
    process_camera_streams(camera_sources, known_embeddings, known_names)

if __name__ == "__main__":
    main()
