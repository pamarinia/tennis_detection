from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # Read Video
    input_video_path = 'input/Med_Djo_cut.mp4'
    video_frames = read_video(input_video_path)

    # Detect Players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detection = player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path='tracker_stubs/player_detection.pkl')

    # Detect Ball
    ball_tracker = BallTracker(model_path='models/yolo5_best.pt')
    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path='tracker_stubs/ball_detection.pkl')
    
    # Court lines detection
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Draw output

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    # Draw ball bounding boxe
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detection)
    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    save_video(video_frames, 'output_videos/Med_Djo_cut.avi')

if __name__ == "__main__":
    main()