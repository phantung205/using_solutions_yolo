from src import config
import cv2
from deploy import inference_sahi
import os
from datetime import datetime

upload_folder = config.upload_folder
result_folder = config.result_folder


def predict_image_sahi(image,model):
    image_name = image.filename

    # save image upload
    name, ext = os.path.splitext(image_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_image_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_image_name)
    image.save(input_path)

    # predict using sahi
    result = inference_sahi.predict_image_sahi(input_path,model)

    # save result
    output_image_name = f"{name}_sahi_{timestamp}"
    result.export_visuals(
        export_dir=result_folder,
        file_name=output_image_name
    )

    return output_image_name + ".png"


def  predict_video_sahi(video, model):
    video_name = video.filename

    # save video upload
    name,ext = os.path.splitext(video_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_video_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder,new_video_name)
    video.save(input_path)

    # đọc video
    cap = cv2.VideoCapture(input_path)

    # kiểm tra xem mở đc video ko
    if not cap.isOpened():
        raise ValueError("lỗi ko mở đc video")

    # lấy ra thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # tạo video lưu kết quả
    output_video_name = f"{name}_sahi_{timestamp}{ext}"
    out_path = os.path.join(result_folder,output_video_name)
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    while True:
        ret,frame = cap.read()

        if not ret:
            break

        result = inference_sahi.predict_image_sahi(frame,model)

        # vẽ bounding box
        for obj in result.object_prediction_list:
            x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())

            label = obj.category.name
            score = obj.score.value
            # vẽ hình vuông
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # vẽ nhãn
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # Lưu frame
        video_writer.write(frame)

    cap.release()
    video_writer.release()

    return output_video_name
