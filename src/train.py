import argparse
import yaml
from src import config
from ultralytics import YOLO,settings
import os


def get_args():
    parser = argparse.ArgumentParser("train model yolo for my dataset")
    # config yaml
    parser.add_argument("--config","-c",type=str,default=config.default_yaml,help="Đường dẫn file YAML cấu hình")

    # parameter
    parser.add_argument("--weights","-w",type=str,default=None,help="đường dẫn tham số model YOLO")
    parser.add_argument("--epochs","-e",type=int,default=None,help="số lượng epochs")
    parser.add_argument("--batch","-b",type=int,default=None,help="số lượng batch size")
    parser.add_argument("--device","-d",type=str,default=None,help="Ghi đè thiết bị (VD: 0 hoặc cpu)")
    parser.add_argument("--workers","-k",type=int,default=None,help="số lượng nhân")
    parser.add_argument("--image_size","-i",type=int,default=None,help="kích thước ảnh")
    parser.add_argument("--learning_rate","-l",type=float,default=None,help="số learning rate")
    parser.add_argument("--weight_decay","-a",type=float,default=None,help="số weight decay")

    args = parser.parse_args()
    return args

def train(args):

    # mặc định hệ thống tạo ra tensorboard
    settings.update({'tensorboard': True})

    # đọc file yamal
    with open(args.config, 'r',encoding='utf-8') as file:
        hyp_config = yaml.safe_load(file)

    # lấy ra đường dẫn file config data mặc định  nếu ko có gán cứng configs/vehicle.yaml
    hyp_config["data"] = os.path.join(config.root_path,hyp_config.get('data', 'configs/vehicle.yaml'))

    if "project" in hyp_config:
        hyp_config["project"] = os.path.join(config.root_path, hyp_config["project"])

    # Override tham số từ CLI
    if args.weights is not None:
        hyp_config["model"] = args.weights
    if args.epochs is not None:
        hyp_config["epochs"] = args.epochs
    if args.batch is not None:
        hyp_config["batch"] = args.batch
    if args.device is not None:
        hyp_config["device"] = int(args.device) if args.device.isdigit() else args.device
    if args.workers is not None:
        hyp_config["workers"] = args.workers
    if args.image_size is not None:
        hyp_config["imgsz"] = args.image_size
    if args.learning_rate is not None:
        hyp_config["lr0"] = args.learning_rate
    if args.weight_decay is not None:
        hyp_config["weight_decay"] = args.weight_decay

    # khởi tạo model
    model_path = hyp_config.get("model","weights/yolov8n.pt")
    if not os.path.isabs(model_path):
        model_path = os.path.join(config.root_path,model_path)

    print("khởi tạo model")
    model = YOLO(model_path)

    print("bắt đầu train")
    model.train(**hyp_config)


if __name__ == '__main__':
    args = get_args()
    train(args)

