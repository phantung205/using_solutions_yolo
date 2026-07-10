from src import config
import cv2
from deploy import inference_sahi
import os
from datetime import datetime

upload_folder = config.upload_folder
result_folder = config.result_folder


def predict_image_sahi(image,model):
    image_name = image.filename

    # save image upload
    name, ext = os.path.splitext(image_name)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_image_name = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_image_name)
    image.save(input_path)

    # predict using sahi
    result = inference_sahi.predict_image_sahi(input_path,model)

    # save result
    output_image_name = f"{name}_prediction_{timestamp}"
    result.export_visuals(
        export_dir=result_folder,
        file_name=output_image_name
    )

    return output_image_name + ".png"