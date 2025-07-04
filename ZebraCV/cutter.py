import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=30):
    os.makedirs(output_folder, exist_ok=True) 
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break   
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Извлечено {saved_count} кадров из {frame_count}")
extract_frames("input.mov", "output_frames", frame_interval=30)