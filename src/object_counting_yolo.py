import cv2

from ultralytics import solutions,YOLO
from src import config
import os

cap = cv2.VideoCapture(config.video_test)
assert cap.isOpened(), "Error reading video file"

# line counting
region_points = [[605, 221], [894, 211], [1243, 389], [799, 427]]  # rectangular region

# check path results
os.makedirs(config.path_result,exist_ok=True)
path_result_video = os.path.join(config.path_result,"object_counting_output.avi")

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(path_result_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model=config.path_model_best,  # model="yolo26n-obb.pt" for object counting with OBB model.
    classes=[1, 2, 3, 4, 5],
    tracker="botsort.yaml",  # choose trackers, e.g., "bytetrack.yaml"
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows