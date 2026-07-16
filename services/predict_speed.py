from src import config
import os
import cv2
from deploy import inference_solution
from  datetime import datetime

upload_folder = config.upload_folder
result_folder = config.result_folder



def predict_video_speed(video,model):
    video_name = video.filename

    # save video upload
    name, ext = os.path.splitext(video_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_video_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_video_name)
    video.save(input_path)

    # đọc video
    cap = cv2.VideoCapture(input_path)

    # kiểm tra xem có đọc đc ko
    if not cap.isOpened():
        raise ValueError("lỗi ko mở đc video")

    # lấy ra thông số của video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # path save video result
    output_video_name = f"{name}_counting_{timestamp}{ext}"
    out_path = os.path.join(result_folder, output_video_name)
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    # load model
    speedestimator = inference_solution.load_speed_model(model, fps)

    while cap.isOpened():
        success, frame = cap.read()

        if not success :
            break

        results = speedestimator(frame)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()

    return output_video_name