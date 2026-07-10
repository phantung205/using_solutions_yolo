from sahi.predict import get_sliced_prediction

def predict_image_sahi(image_path,model):
    result = get_sliced_prediction(
        image=image_path,
        detection_model=model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,

    )

    return result