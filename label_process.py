import cv2
import numpy as np
import os

def label_process(Ground_path, label_path, label_num):
    img = cv2.imread(Ground_path)
    nr, nc = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # 尋找輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        with open(label_path + ".txt", 'a') as f:
            for contour in contours:
                ##忽略過小的出血點
                if label_num == 1:
                    if cv2.contourArea(contour) < 100:  # 面積小於50的忽略
                        continue

                x, y, w, h = cv2.boundingRect(contour)  
                x_center = (x + w / 2) / nc
                y_center = (y + h / 2) / nr
                width = w / nc
                height = h / nr
                f.write(f"{label_num} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}\n")
    else:
        open(label_path + ".txt", 'a').close()

#############  main  #############
folder_Ground= r"dataset\IDRiD\A. Segmentation\Groundtruths"
for i in range(2):
    if i == 0:
        status = "train"
    else:
        status = "val"
    folder_Ground_path = os.path.join(folder_Ground, status)

    all_lesions = os.listdir(folder_Ground_path)
    for lesion_name in all_lesions:
        lesion_path = os.path.join(folder_Ground_path, lesion_name)
        files = os.listdir(lesion_path)
        for file in files:
            Ground_img = os.path.join(lesion_path, file)
            lesion_file = os.path.splitext(file)[0]
            # print(lesion_file, lesion_name, status)

            if lesion_name == "MA":
                label_num = 0
            elif lesion_name == "HE":
                label_num = 1
            elif lesion_name == "EX":
                label_num = 2
            elif lesion_name == "SE":
                label_num = 3
            else:
                label_num = 4
            
            folder_label = r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\labels"
            folder_label = os.path.join(folder_label, status)
            label_path = os.path.join(folder_label, lesion_file)
            # print(Ground_img, label_path)
            label_process(Ground_img, label_path, label_num)
print("✅ 標註檔案處理完成！")

