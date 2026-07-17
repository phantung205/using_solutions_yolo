# Detect Vehicle (Using the YOLO object detection model and YOLO's solution.)

Ứng dụng học sâu (Deep learning) vào bài toán phạt hiện đối tượng đi là xe cộ trong thời gian thực tế 
tại các camera và sử dụng các giải pháp có sẵn của yolo .tích hợp vào wed, sử dụng flask 

---

## 1. chức năng 
- dữ đoạn ảnh hoặc video có những đối tượng xe cộ nào
- sử dụng các giải pháp :
  - đếm số lượng xe trong một vùng
  - đếm số lượng xe trong nhiều vung khác nhau trên đường
  - biểu đồ nhiệt cho thấy vị trí đậm mầu thường xuyên suất hiện đối tượng
  - object tracking
  - đo tốc độ của các phương tiện
- bao gồm 5 class :
  - vehicles
  - bicycle
  - bus
  - car
  - motorbike
  - truck

## 2. kết quả 
### 2.1 trong quá trình train
![Tensorboard result](data/img.png)

### 2.2 một số kêt quả của sử dụng giải pháp

https://github.com/user-attachments/assets/3a15fc6c-e9e8-47fc-ba26-5f3a91972bdc

### 2.3 hình ảnh giao diện wed

![Interface wed](data/img_1.png)


---

## 3. Cấu trúc thư mục

```text
helmet_detection/
├── configs
├── data
│   ├── processed
│   │   ├── images
│   │   │   ├── train
│   │   │   └── valid
│   │   └── labels
│   │       ├── train
│   │       └── valid
│   ├── raw
│   │   ├── test
│   │   ├── train
│   │   └── valid
│   └── test
├── deploy
├── result
│   ├── predict_base
│   ├── predict_solution
│   └── traffic_detector
│       └── weights
├── routes
├── services
├── src
├── static
│   ├── results
│   └── uploads
├── templates
├── using_sahi
├── using_solution_yolo
└── weights
```

---

## 4 Dataset

### 4.1 Tải dữ liệu

- link tải dữ liệu:  
 https://universe.roboflow.com/traffic-camera/vehicles-22g3b/dataset/11


### 4.2 Cách dùng dữ liệu
### 1.Đặt vào thư mục:
```text
data/raw
      ├── test
      ├── train
      └── valid
```
### 2. chia dữ liệu thành đùng dữ liệu đang cần
```bash
python -m src.prepare_data
```

--- 

## 5. Cài đặt

### 5.1 Tạo môi trường ảo (khuyên dùng)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 5.2 Cài thư viện

```bash
pip install -r requirements.txt
```

---

## 6. chỉnh cấu hình tham số mặc định
### 6.1 cấu hình đường dẫn
```text
src/config.py
```

### 6.2 cấu hình các tham số model
```text
configs/train_hyp.yaml
```

---

## 7 train
### 7.1 chạy các lệnh sau

```bash
python -m src.train 
```

### 7.2 kết qủa checkpoint sau khi train lưu trong:
```text
result/
```

---

## 8. chạy docker file
### 1. build docker image
```bash
docker build -t vehicle .
```

### 2. vào trong docker image
```bash
docker run -it --rm --gpus all -v ${PWD}/data/:/work/data  -v ${PWD}/result:/work/result  vehicle  bash
```

- sau khi vào trong docker containner chạy lệnh để train model trong docker:
```bash
python -m src.train 
```

### 3. chạy luôn wed ko cần vào docker, nếu đã có checkpoint train rồi
```bash
docker run --rm -p 5000:5000 --gpus all -v ${PWD}/result/traffic_detector:/work/result/traffic_detector  -v ${PWD}/static:/work/static  vehicle  
```

---

## 9. xem quá trình train và test thử ở local

### 1. xem quá trình train
```bash
tensorboard --logdir result/traffic_detector 
```
### 2. test thử local
```text
python -m src.inference -i (đường dẫn ảnh)
python -m src.inference -v (đường dẫn video)
```
### 3. các giải pháp của yolo cung cấp local
- đếm số lượng xe trong một vùng:
```bash
python -m src.object_counting_yolo
```

- đếm số lượng xe trong nhiều vung khác nhau trên đường:
```bash
python -m object_counting_region_yolo
```

- biểu đồ nhiệt cho thấy vị trí đậm mầu thường xuyên suất hiện đối tượng:
```bash
python -m Heatmaps_yolo
```

- đo tốc độ của các phương tiện:
```bash
python -m speed_estimation_yolo
```

- kết quả sẽ đc lưu trong folder results

---

## 10. chạy wed

```bash
python app.py
```
Mặc định ứng dụng chạy tại:

http://127.0.0.1:5000

