import cv2
from src import config
from ultralytics import solutions
import os


cap = cv2.VideoCapture(config.video_test)
assert cap.isOpened(), "Error reading video file"

# check path results
os.makedirs(config.path_result,exist_ok=True)
path_result_video = os.path.join(config.path_result,"speed_management.avi")


# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(path_result_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize speed estimation object
speedestimator = solutions.SpeedEstimator(
    show=True,  # display the output
    model=config.path_model_best,  # path to the YOLO26 model file.
    fps=fps,  # adjust speed based on frame per second
    # max_speed=120,  # cap speed to a max value (km/h) to avoid outliers
    # max_hist=5,  # minimum frames object tracked before computing speed
    meter_per_pixel=0.10,  # highly depends on the camera configuration
    # classes=[0, 2],  # estimate speed of specific classes.
    # line_width=2,  # adjust the line width for bounding boxes
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = speedestimator(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()