import  os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir =  os.path.join(root_path,"data")
data_raw_dir = os.path.join(data_dir,"raw")
data_preprocessing_dir = os.path.join(data_dir,"processed")
# video test
video_test = os.path.join(data_dir,"test","video1.mp4")

splits=["train","valid"]

categoris = ["vehicles","bicycle","bus","car","motorbike","truck"]

default_yaml = os.path.join(root_path,"configs","train_hyp.yaml")

# path model
model_dir = os.path.join(root_path,"result","traffic_detector","weights")
path_model_best = os.path.join(model_dir,"best.pt")


#path save result solution
path_result = os.path.join(root_path,"result","solution")

