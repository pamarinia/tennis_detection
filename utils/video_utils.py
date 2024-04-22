import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()