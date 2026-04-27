import cv2
from src import config
import os
from ultralytics import solutions

cap = cv2.VideoCapture(config.video_test)
assert cap.isOpened(), "Error reading video file"

# Pass region as list
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Pass region as dictionary
region_points = {
    "region-01": [[1212, 384], [1137, 432], [764, 197], [817, 178]],
    "region-02": [[1137, 431], [991, 447], [689, 217], [753, 204]],
    "region-03": [[592, 227], [688, 221], [981, 447], [826, 447]],
    "region-04": [[368, 258], [482, 247], [622, 498], [418, 509]],
    "region-05":[[248, 260], [360, 256], [416, 509], [218, 507]],
    "region-06": [[111, 253], [233, 266], [195, 503], [8, 447]],
    "region-07":[[25, 460], [233, 526], [639, 513], [832, 509], [1045, 522], [1043, 700], [81, 706]],
}

# check path results
os.makedirs(config.path_result,exist_ok=True)
path_result_video = os.path.join(config.path_result,"region_counting.avi")


# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(path_result_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize region counter object
regioncounter = solutions.RegionCounter(
    show=True,  # display the frame
    region=region_points,  # pass region points
    model=config.path_model_best,  # model for counting in regions, e.g., yolo26s.pt
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = regioncounter(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows