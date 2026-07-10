
def predict_base(model, image_path):
    results = model(image_path)

    annotated_image = results[0].plot()

    return annotated_image

