from ultralytics import YOLO
from sahi import AutoDetectionModel

base_model = None
sahi_model = None



def load_base_model(checkpoint):
    global base_model

    if base_model is None:
        base_model = YOLO(checkpoint)

    return base_model


def load_sahi_model(checkpoint):
    global sahi_model

    if sahi_model is None:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=checkpoint,
            confidence_threshold=0.7,
            device="cuda:0"
        )

    return sahi_model



