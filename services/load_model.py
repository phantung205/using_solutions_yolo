from deploy import load_model_yolo

yolo_model = None
sahi_model = None

def load_models(checkpoints):
    global yolo_model, sahi_model

    if yolo_model is None:
        yolo_model = load_model_yolo.load_model_base(checkpoints)

    if sahi_model is None:
        sahi_model = load_model_yolo.load_model_sahi(checkpoints)

    return yolo_model, sahi_model