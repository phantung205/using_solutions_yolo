import  os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir =  os.path.join(root_path,"data")
data_raw_dir = os.path.join(data_dir,"raw")
data_preprocessing_dir = os.path.join(data_dir,"preprocessing")
splits=["train","valid"]

categoris = ["vehicles","bicycle","bus","car","motorbike","truck"]