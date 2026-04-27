import cv2

from ultralytics import solutions,YOLO
from src import config
import os

cap = cv2.VideoCapture(config.video_test)
assert cap.isOpened(), "Error reading video file"

# check path results
os.makedirs(config.path_result,exist_ok=True)
path_result_video = os.path.join(config.path_result,"heatmap_output.avi")

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(path_result_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# For object counting with heatmap, you can pass region points.
# region_points = [(20, 400), (1080, 400)]                                      # line points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon points

# Initialize heatmap object
heatmap = solutions.Heatmap(
    show=True,  # display the output
    model=config.path_model_best,  # path to the YOLO26 model file
    colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
    # region=region_points,  # object counting with heatmaps, you can pass region_points
    # classes=[0, 2],  # generate heatmap for specific classes, e.g., person and car.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = heatmap(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()