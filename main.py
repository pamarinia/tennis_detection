from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # Read Video
    input_video_path = 'input/Med_Djo_cut.mp4'
    video_frames = read_video(input_video_path)

    # Detect Players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detection.pkl')

    # Detect Ball
    ball_tracker = BallTracker(model_path='models/yolo5_best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/ball_detection.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court lines detection
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_player(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Draw output

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    # Draw ball bounding boxe
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    save_video(output_video_frames, 'output_videos/Med_Djo_cut.avi')

if __name__ == "__main__":
    main()