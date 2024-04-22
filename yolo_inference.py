from ultralytics import YOLO
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('yolov8x')  # Load model

# Cut the first 10 seconds of the video
ffmpeg_extract_subclip('input\Medvedev_Djokovic_2023_US_Open_Final.mp4', 178, 200, targetname='input\Med_Djo_cut.mp4')

# Now use the clipped video
result = model.track('input\Med_Djo_cut.mp4', conf=0.2, save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)