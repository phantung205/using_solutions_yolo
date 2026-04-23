import os
from src import config
import json
from pprintpp import pprint
import shutil
import cv2


def main():
    for split in config.splits:
        # tạo thư mục train val trong processing
        dir_images = os.path.join(config.data_preprocessing_dir,"images",split)
        dir_vals = os.path.join(config.data_preprocessing_dir,"labels",split)

        if os.path.isdir(dir_images):
            shutil.rmtree(dir_images)
        if os.path.isdir(dir_vals):
            shutil.rmtree(dir_vals)

        os.makedirs(dir_images)
        os.makedirs(dir_vals)

        # lấy tất cả danh sách trong thư mục
        list_image = []
        list_label = []
        dir_raw_data = os.path.join(config.data_raw_dir,split)
        for file in os.listdir(dir_raw_data):
            # sử lý và lưu ảnh
            if file.lower().endswith((".jpg", ".png")):
                list_image.append(file)
                src = os.path.join(dir_raw_data,file)
                dst = os.path.join(dir_images,file)
                shutil.copy(src, dst)

            # sử lỹ nhãn và lưu nhãn
            if file.lower().endswith((".json")):
                with open(os.path.join(dir_raw_data,file), 'r') as f:
                    data = json.load(f)
                    for img in data["images"]:
                        list_label.append(img["file_name"])
                        for anno in data["annotations"]:
                            if(img["id"] == anno["image_id"]):
                                xmin, ymin, w, h = anno["bbox"]
                                img_w = img["width"]
                                img_h = img["height"]
                                # center
                                x_ct = xmin + w / 2
                                y_ct = ymin + h / 2
                                # normalize
                                x_ct /= img_w
                                y_ct /= img_h
                                w /= img_w
                                h /= img_h

                                label_name = os.path.splitext(img["file_name"])[0] + ".txt"
                                path_labels = os.path.join(dir_vals,label_name)
                                with open(path_labels, 'a') as file_text:
                                    file_text.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(anno["category_id"]-1,x_ct,y_ct,w,h))
        # name image
        image_names = {
            os.path.splitext(f)[0]
            for f in os.listdir(dir_images)
        }
        # name label
        label_names = {
            os.path.splitext(f)[0]
            for f in os.listdir(dir_vals)
        }

        # xóa image nếu ko có label
        for img in os.listdir(dir_images):
            name = os.path.splitext(img)[0]
            if name not in label_names:
                os.remove(os.path.join(dir_images, img))


        # xóa label nếu ko có ảnh
        for label in os.listdir(dir_vals):
            name = os.path.splitext(label)[0]
            if name not in image_names:
                os.remove(os.path.join(dir_vals, label))

        print(len(os.listdir(dir_images)))
        print(len(os.listdir(dir_vals)))

if __name__ == '__main__':
    main()
