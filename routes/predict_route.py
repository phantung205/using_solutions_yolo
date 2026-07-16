from flask import Blueprint,render_template,request
from services import validation,predict_base,load_model,predict_sahi,predict_heatmap,predict_counting,predict_counting_region,predict_speed
from src import config
import os

predict_bp = Blueprint("predict",__name__)

# Load model khi import file này
base_model, sahi_model = load_model.load_models(config.path_model_best)


@predict_bp.route("/")
def home():
    return render_template("index.html",result=None,error=None)


@predict_bp.route("/predict",methods=["POST"])
def predict():
    try:
        # nhận thông tin form
        file = request.files["file"]
        solution = request.form["solution"]

        # validation
        validation.validate_ext(file)

        # lấy ra đuôi định dạng thêm thuộc loại nào
        ext = os.path.splitext(file.filename)[1].lower()

        # kiểm tra xem là ảnh hay là video
        image_extensions = {".jpg",".jpeg",".png",".bmp"}
        if ext in image_extensions:
            if solution == "base":
                image_name = predict_base.predict_image_base(file,base_model)

            elif solution == "sahi":
                image_name = predict_sahi.predict_image_sahi(file,sahi_model)

            elif solution == "heatmap":
                image_name = predict_heatmap.predict_image_heatmap(file,base_model)

            elif solution == "counting":
                image_name = predict_counting.predict_image_counting(file,base_model)

            elif solution == "counting_region":
                image_name = predict_counting_region.predict_image_counting_region(file,base_model)

            elif solution == "speed":
                raise ValueError(" ko hỗi trợ hình ảnh đo tốc độ cách sử lý này")

            else:
                raise ValueError("lỗi ko hỗi trợ cách sử lý này")

            return render_template("index.html", result=f"results/{image_name}", error=None)

        else :
            if solution == "base":
                video_name = predict_base.predict_video_base(file,base_model)

            elif solution == "sahi":
                video_name = predict_sahi.predict_video_sahi(file,sahi_model)

            elif solution == "heatmap":
                video_name = predict_heatmap.predict_video_heatmap(file,base_model)

            elif solution == "counting":
                video_name = predict_counting.predict_video_counting(file,base_model)

            elif solution == "counting_region":
                video_name = predict_counting_region.predict_video_counting_region(file,base_model)

            elif solution == "speed":
                video_name = predict_speed.predict_video_speed(file,base_model)

            else:
                raise ValueError("lỗi ko hỗi trợ cách sử lý này")

            return render_template("index.html", result=f"results/{video_name}", error=None)



    except Exception as e:
        return render_template("index.html",result=None,error=str(e))