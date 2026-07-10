from ultralytics import YOLO
from sahi import AutoDetectionModel

model_base = None
model_sahi = None

def load_model_base(checkpoint):
    global model_base

    if model_base is None:
        model_base = YOLO(checkpoint)

    return model_base



def load_model_sahi(checkpoint):
    global model_sahi

    if model_sahi is None:
        model_sahi = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=checkpoint,
            confidence_threshold=0.7,
            device="cuda:0"
        )

    return model_sahi