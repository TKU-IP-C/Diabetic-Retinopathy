import cv2
import os
import numpy as np

def img_process(img_path, img_num, status, lesion_name):
    ## 切割影像 ##
    img = cv2.imread(img_path, 0)
    nr, nc = img.shape[:2]
    num_list = [0, 1, 2, 3, 4] # 0: original, 1: left_up, 2: right_up, 3: left_down, 4: right_down
    for num in num_list:
        ROI_left_up = img[0:int(nr//2), 0:int(nc//2)]
        ROI_right_up = img[0:int(nr//2), int(nc//2):nc]
        ROI_left_down = img[int(nr//2):nr, 0:int(nc//2)]
        ROI_right_down = img[int(nr//2):nr, int(nc//2):nc]
        if num == 0:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/Groundtruths/{status}/{lesion_name}/IDRiD_{img_num}_{num}.tif", img)
        if num == 1:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/Groundtruths/{status}/{lesion_name}/IDRiD_{img_num}_{num}.tif", ROI_left_up)
        if num == 2:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/Groundtruths/{status}/{lesion_name}/IDRiD_{img_num}_{num}.tif", ROI_right_up)
        if num == 3:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/Groundtruths/{status}/{lesion_name}/IDRiD_{img_num}_{num}.tif", ROI_left_down)
        if num == 4:
            cv2.imwrite(f"dataset/IDRiD/A. Segmentation/Groundtruths/{status}/{lesion_name}/IDRiD_{img_num}_{num}.tif", ROI_right_down)


#### main ####
folder = r"dataset\IDRiD\A. Segmentation\2. All Segmentation Groundtruths"
for i in range(5):
    if i == 0:
        lesion = "1. Microaneurysms"
    elif i == 1:
        lesion = "2. Haemorrhages"
    elif i == 2:
        lesion = "3. Hard Exudates"
    elif i == 3:
        lesion = "4. Soft Exudates"
    else:
        lesion = "5. Optic Disc"
    for j in range(2):
        if j == 0:
            folder_path = os.path.join(folder, f"a. Training Set", lesion)
            status = "train"
        else:
            folder_path = os.path.join(folder, f"b. Testing Set", lesion)
            status = "val"
        for file in os.listdir(folder_path):
            file_name = os.path.splitext(file)[0]
            img_num = file_name.split("_")[-2]
            lesion_name = file_name.split("_")[-1]
            img_process(os.path.join(folder_path, file), img_num, status, lesion_name)