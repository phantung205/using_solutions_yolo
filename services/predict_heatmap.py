from deploy import inference_solution
from datetime import datetime
import os
from src import config
import cv2

upload_folder = config.upload_folder
result_folder = config.result_folder

def predict_image_heatmap(image, model):
    image_name = image.filename

    # Lưu ảnh upload
    name, ext = os.path.splitext(image_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_image_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_image_name)
    image.save(input_path)

    # Đọc ảnh
    image = cv2.imread(input_path)

    # Khởi tạo Heatmap
    heatmap_model = inference_solution.load_heatmap_model(model)

    # Dự đoán
    result_frame = heatmap_model(image)

    # Lưu kết quả
    output_image_name = f"{name}_heatmap_{timestamp}.jpg"
    output_path = os.path.join(result_folder, output_image_name)

    success = cv2.imwrite(output_path, result_frame)

    if not success:
        raise ValueError("Không thể lưu ảnh kết quả")

    return output_image_name


def predict_video_heatmap(video,model):
    video_name = video.filename

    #save video upload
    name, ext = os.path.splitext(video_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_video_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_video_name)
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

    # path save video result
    output_video_name = f"{name}_heatmap_{timestamp}{ext}"
    out_path = os.path.join(result_folder, output_video_name)
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    # Initialize heatmap object
    heatmap_model = inference_solution.load_heatmap_model(model)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        results = heatmap_model(frame)

        video_writer.write(results.plot_im)  # write the processed frame.

    cap.release()
    video_writer.release()

    return output_video_name