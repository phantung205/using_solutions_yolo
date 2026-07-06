import argparse
import cv2
from src import config
import os
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser("infence model yolo for vehicel")
    parser.add_argument("--image_path","-i",type=str,default=config.image_test,help="đường dẫn ảnh")
    parser.add_argument("--video_path","-v",type=str,default=config.video_test,help="đường dẫn video")
    parser.add_argument("--checkpoint","-c",type=str,default=config.path_model_best,help="đường dẫn checkppoint model")

    args = parser.parse_args()
    return args

def predict_image(image_path , model):
    image = cv2.imread(image_path)

    results = model(image)

    annotated_image = results[0].plot()

    # save image
    path_save = os.path.join(config.path_predict,"result_image.jpg")
    cv2.imwrite(path_save,annotated_image)
    print("kết quả đã đc lưu tại file: ",path_save)

    #show image
    cv2.imshow("result", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_video(video_path , model):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    # lấy ra thông số của video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_path =  os.path.join(config.path_predict,"result_video.mp4")
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)

        # vẽ box
        annotated_frame = results[0].plot()

        # Lưu vào video mới
        video_writer.write(annotated_frame)

        cv2.imshow("YOLO Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("kết quả đã đc lưu tại file: ", out_path)


def main():
    args = get_args()

    # load model
    model = YOLO(args.checkpoint)

    # predict image
    if args.image_path is not None:
        if not os.path.exists(args.image_path):
            print("ko tìm thấy ảnh ")
            return
        predict_image(args.image_path,model)

    # predict video
    if args.video_path is not None:
        if not os.path.exists(args.video_path):
            print("ko tìm thấy ảnh ")
            return
        predict_video(args.video_path, model)

    if args.image_path is not None and args.video_path is not None:
        print("bạn chưa truyển ảnh hoặc video test vào")
if __name__ == '__main__':
    main()