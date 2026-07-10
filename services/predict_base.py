from src import config
import os
from datetime import datetime
from deploy import inference_base
import cv2

upload_folder = config.upload_folder
result_folder = config.result_folder


def predict_image_base(image,model):
    image_name = image.filename

    # save image
    name, ext = os.path.splitext(image_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_image_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_image_name)
    image.save(input_path)

    # predict image
    result_image = inference_base.predict_base(model, input_path)

    # save result
    output_image_name = f"{name}_prediction_{timestamp}.jpg"
    output_path = os.path.join(result_folder, output_image_name)

    success = cv2.imwrite(output_path, result_image)

    if not success:
        raise ValueError("Không thể lưu ảnh kết quả")

    return output_image_name


def predict_video_base(video,model):
    video_name = video.filename

    # save video input
    name,ext = os.path.splitext(video_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_video_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder,new_video_name)

    video.save(input_path)

    # predict video
    cap = cv2.VideoCapture(input_path)

    # kiểm tra xem mở đc video ko
    if not cap.isOpened():
        raise ValueError("Không thể mở video")

    # lấy ra thông số của video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #kiểm tra fbs
    if fps <= 0:
        fps = 30

    # name video result
    output_video_name = f"{name}_result_{timestamp}{ext}"
    out_path = os.path.join(config.result_folder, output_video_name)
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        result_frame = inference_base.predict_base(model, frame)

        video_writer.write(result_frame)

    cap.release()
    video_writer.release()

    return output_video_name