from services.crowd_monitor import CrowdMonitor

if __name__ == "__main__":
    monitor = CrowdMonitor(threshold=10)
    monitor.process_stream(video_source=0)  # 0 = webcam, or path to video
