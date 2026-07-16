from deploy import load_model_yolo

def load_models(checkpoint):

    base_model = load_model_yolo.load_base_model(checkpoint)
    sahi_model = load_model_yolo.load_sahi_model(checkpoint)


    return base_model, sahi_model