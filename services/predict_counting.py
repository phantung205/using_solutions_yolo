from deploy import inference_solution
import cv2
import os
from src import config
from datetime import datetime

upload_folder = config.upload_folder
result_folder = config.result_folder

def predict_image_counting(image, model,region_points=None):
    image_name = image.filename

    # save image upload
    name, ext = os.path.splitext(image_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_image_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_image_name)
    image.save(input_path)


    # đọc ảnh
    image  = cv2.imread(input_path)

    # khởi tạo model counting
    counter = inference_solution.load_counting_model(model,region_points)

    # dự đoán
    result_image = counter(image)

    # lưu kết quả dự đoán
    output_image_name = f"{name}_counting_{timestamp}.jpg"
    output_path = os.path.join(result_folder, output_image_name)

    success = cv2.imwrite(output_path, result_image.plot_im)

    if not success:
        raise ValueError("Không thể lưu ảnh kết quả")

    return output_image_name


def predict_video_counting(video,model,region_points=None):
    video_name = video.filename

    # save video upload
    name, ext = os.path.splitext(video_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_video_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_video_name)
    video.save(input_path)

    # đọc video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("lỗi ko mở đc video")

        # lấy ra thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # path save video result
    output_video_name = f"{name}_counting_{timestamp}{ext}"
    out_path = os.path.join(result_folder, output_video_name)
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    # khởi tạo model
    counting_model = inference_solution.load_counting_model(model,region_points)

    while cap.isOpened():
        success , frame = cap.read()

        if not success:
            break

        results = counting_model(frame)

        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()

    return output_video_name