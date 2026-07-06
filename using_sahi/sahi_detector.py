from sahi import AutoDetectionModel
from src import config
from sahi.predict import get_sliced_prediction
import os
import cv2
def load_model():
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=config.path_model_best,
        confidence_threshold=0.7,
        device="cuda:0"
    )

    return  detection_model

# Load model ngay khi chạy chương trình
detection_model = load_model()

def predict_image_sahi(image_path=config.image_test,
                       save_folder=config.path_predict,
                       image_name="result_predict_sahi"):
    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,

    )

    result.export_visuals(
        export_dir=save_folder,
        file_name=image_name
    )
    path_show = os.path.join(save_folder,f"{image_name}.png")
    img = cv2.imread(path_show)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_video_sahi(video_path=config.video_test,
                       folder_save=config.path_predict,
                       name_video="video_sahi.mp4"):
    #đọc video
    cap = cv2.VideoCapture(video_path)

    # lấy thống tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # tạo video đẻ lưu kết quả
    save_path = os.path.join(folder_save,name_video)
    writer = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = get_sliced_prediction(
            image=frame,
            detection_model=detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # vẽ bounding box
        for obj in result.object_prediction_list:
            x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())

            label = obj.category.name
            score = obj.score.value
            # vẽ hình vuông
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # vẽ nhãn
            cv2.putText(frame,f"{label} {score:.2f}",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)

        cv2.imshow("Video", frame)

        # Lưu frame
        writer.write(frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # predict_image_sahi()
    predict_video_sahi()