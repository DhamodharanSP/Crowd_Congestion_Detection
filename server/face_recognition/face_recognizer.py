import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from utils.cloudinary_uploader import upload_frame
from utils.twilio_notifier import send_whatsapp_message

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Init models
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ---- Helpers ----
def extract_face(image, box):
    x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face_resized = cv2.resize(face, (160, 160))
    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return face_tensor.to(device)


def compute_embedding(face_tensor):
    with torch.no_grad():
        return resnet(face_tensor).cpu().numpy().flatten()


# ---- Load known faces ----
def load_known_faces(folder_path="known_faces"):
    known_embeddings, known_names = [], []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes, _ = mtcnn.detect(rgb)
            if boxes is not None:
                face = extract_face(rgb, boxes[0])
                if face is not None:
                    emb = compute_embedding(face)
                    if emb is not None:
                        known_embeddings.append(emb)
                        known_names.append(os.path.splitext(filename)[0])
    return known_embeddings, known_names


# ---- Recognize face ----
def recognize_face(frame, known_embeddings, known_names, threshold=0.7):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for box in boxes:
            face_tensor = extract_face(rgb, box)
            if face_tensor is not None:
                emb = compute_embedding(face_tensor)
                sims = [cosine_similarity([emb], [ke])[0, 0] for ke in known_embeddings]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]

                if best_score > threshold:
                    return known_names[best_idx], best_score, box
    return None, None, None


# ---- Main FaceRecognizer class ----
class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_embeddings, self.known_names = load_known_faces(known_faces_dir)
        print(f"Loaded {len(self.known_names)} known faces")

    def process_stream(self, video_source=0, camera_id=0, db=None):
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            name, score, box = recognize_face(frame, self.known_embeddings, self.known_names)

            if name:
                # Draw bounding box
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Always log recognition step first
                print(f"Recognized {name} with score {score:.2f} in camera {camera_id}")

                # Save + upload full frame
                tmp = f"{name}_det.jpg"
                cv2.imwrite(tmp, frame)
                cloud_url = upload_frame(tmp, "Crowd_Detection")
                os.remove(tmp)

                if db and cloud_url:
                    record = {
                        "name": name,
                        "score": float(score),
                        "camera_id": camera_id,
                        "url": cloud_url
                    }
                    db.detections.insert_one(record)
                    print(f"✅ Stored {name}'s detection in MongoDB with Cloudinary URL: {cloud_url}")

                # Send WhatsApp alert
                if cloud_url:
                    sid = send_whatsapp_message(cloud_url, f"⚠️ Recognized {name} ({score:.2f}) in camera {camera_id}")
                    if sid:
                        print(f"WhatsApp message sent: {sid}")

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
