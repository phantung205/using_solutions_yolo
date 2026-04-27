import cv2
from src import config
import os

cap = cv2.VideoCapture(config.video_test)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    path_save_image = os.path.join(config.data_dir,"test","image_one_frame.jpg")
    cv2.imwrite(path_save_image, frame)

    break

# Release resources
cap.release()
cv2.destroyAllWindows()