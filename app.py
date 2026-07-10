from flask import Flask
from routes.predict_route import predict_bp
from src import config
import os

os.makedirs(config.upload_folder, exist_ok=True)
os.makedirs(config.result_folder, exist_ok=True)


app = Flask(__name__)



app.register_blueprint(predict_bp)


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )