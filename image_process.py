import cv2
import os
import numpy as np

def img_process(img, img_num, status):
    gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(5, 5))
    enhanced_gray = clahe.apply(gray)

    ## 切割影像 ##
    nr, nc = enhanced_gray.shape[:2]
    num_list = [0, 1, 2, 3, 4] # 0: original, 1: left_up, 2: right_up, 3: left_down, 4: right_down
    for num in num_list:
        ROI_left_up = enhanced_gray[0:int(nr//2), 0:int(nc//2)]
        ROI_right_up = enhanced_gray[0:int(nr//2), int(nc//2):nc]
        ROI_left_down = enhanced_gray[int(nr//2):nr, 0:int(nc//2)]
        ROI_right_down = enhanced_gray[int(nr//2):nr, int(nc//2):nc]
        if num == 0:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/IDRiD_yolo/images/{status}/IDRiD_{img_num}_{num}.jpg",enhanced_gray)
        if num == 1:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/IDRiD_yolo/images/{status}/IDRiD_{img_num}_{num}.jpg", ROI_left_up)
        if num == 2:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/IDRiD_yolo/images/{status}/IDRiD_{img_num}_{num}.jpg", ROI_right_up)
        if num == 3:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/IDRiD_yolo/images/{status}/IDRiD_{img_num}_{num}.jpg", ROI_left_down)
        if num == 4:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/IDRiD_yolo/images/{status}/IDRiD_{img_num}_{num}.jpg", ROI_right_down)


#### main ####
folder = r"dataset\IDRiD\A. Segmentation\1. Original Images"
for i in range(2):
    if i == 0:
        folder_path = os.path.join(folder, f"a. Training Set")
        status = "train"
    else:
        folder_path = os.path.join(folder, f"b. Testing Set")
        status = "val"
    for file in os.listdir(folder_path):
        file_name = os.path.splitext(file)[0]
        img_num = file_name.split("_")[-1]
        img_process(os.path.join(folder_path, file), img_num, status)
    
# img_path = r"dataset\IDRiD\A. Segmentation\1. Original Images\a. Training Set\IDRiD_01.jpg"

# img_process(img_path)