from face_recognition.face_recognizer import FaceRecognizer

if __name__ == "__main__":
    fr = FaceRecognizer(known_faces_dir="known_faces")
    fr.process_stream(video_source=0)  # webcam or video path
