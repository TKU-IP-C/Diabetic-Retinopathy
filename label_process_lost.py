import cv2
import numpy as np
import os
import re

def Image_turn_to_yolo(img, nc, nr, output_txt_path, label):
    # 轉為灰階並二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 尋找輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # 建立並寫入 txt 檔案
        with open(output_txt_path, 'a') as f:
                
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h < 100:  # 忽略過小的物件
                    continue
                # 計算 YOLO 格式座標 (歸一化)
<<<<<<< HEAD:image_process.py
=======
<<<<<<< HEAD:label_process_lost.py
<<<<<<< HEAD:label_process_lost.py
<<<<<<< HEAD:label_process_lost.py
                x_center = (x + w / 2) / nr
                y_center = (y + h / 2) / nc
                width = w / nr
                height = h / nc
=======
=======
>>>>>>> 40e592e4cb8043624cc2ec7cc6e1c917896aa152:image_process.py
=======
>>>>>>> 40e592e4cb8043624cc2ec7cc6e1c917896aa152:image_process.py
>>>>>>> 80e058a20 (1203):label_process_lost.py
                x_center = max(0, min(1, (x + w / 2) / nr))
                y_center = max(0, min(1, (y + h / 2) / nc))
                width = max(0, min(1, w / nr))
                height = max(0, min(1, h / nc))

>>>>>>> 40e592e4cb8043624cc2ec7cc6e1c917896aa152:image_process.py
                # 寫入檔案 (YOLO 格式)
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    else:
        open(output_txt_path, 'w').close()
    
def read_folder_files(folder_path):
    """
    讀取資料夾中的所有檔案
    """
    # 檢查資料夾是否存在
    if not os.path.exists(folder_path):
        print(f"資料夾不存在: {folder_path}")
        return []
    
    # 獲取所有檔案和資料夾名稱
    all_items = os.listdir(folder_path)
    
    # 只保留檔案（排除子資料夾）
    files_only = []
    for item in all_items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            files_only.append(item)
    
    return files_only

#############  main  #############

allimgPath = r"dataset\IDRiD\A. Segmentation\2. All Segmentation Groundtruths"
# 列出指定路徑底下所有檔案(包含資料夾)
FileList = os.listdir(allimgPath)
for file in FileList:
    if file == "a. Training Set":
        pre_folder_path = os.path.join(allimgPath, file)
        # print(folder_path)
        imglist = os.listdir(pre_folder_path)
        label = 0
        for img in imglist:
            folder_path = os.path.join(pre_folder_path, img)
            # print(folder_path)
            output_folder = r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\labels\train"

            # 確保輸出資料夾存在
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                # print(f"建立輸出資料夾: {output_folder}")

            files = read_folder_files(folder_path)

            # print(f"找到 {len(files)} 個檔案:")

            for file in files:
                # 輸入圖片路徑 - 正確的寫法
                img_path = os.path.join(folder_path, file)
                
                # 輸出 txt 檔案路徑 - 正確的寫法
                # 取得檔案名稱（不含副檔名）
                file_name_without_ext = os.path.splitext(file)[0]
                file_name_without_ext = re.sub(r'_(MA|HE|EX|SE|OD)$', '', file_name_without_ext)

                output_txt_path = os.path.join(output_folder, f"{file_name_without_ext}.txt")
            
                
                # print(f"處理: {file} -> {output_txt_path}")
                
                # 讀取圖片
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"找不到圖片: {img_path}")
                    # 如果找不到圖片，仍然建立空的 txt 檔案
                    open(output_txt_path, 'w').close()
                    print(f"建立空的 txt 檔案: {output_txt_path}")
                else:
                    # 獲取圖片尺寸
                    nc, nr = img.shape[:2]
                    # print(f"圖片尺寸: {nc} x {nr}")
                    
                    # 轉換為 YOLO 格式並保存為 txt
                    Image_turn_to_yolo(img, nc, nr, output_txt_path, label)
            label += 1
    else:
        pre_folder_path = os.path.join(allimgPath, file)
        # print(folder_path)
        imglist = os.listdir(pre_folder_path)
        label = 0
        for img in imglist:
            folder_path =os.path.join(pre_folder_path, img)
            # print(folder_path)
            output_folder = r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\labels\val"

            # 確保輸出資料夾存在
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                # print(f"建立輸出資料夾: {output_folder}")

            files = read_folder_files(folder_path)

            # print(f"找到 {len(files)} 個檔案:")

            for file in files:
                # 輸入圖片路徑 - 正確的寫法
                img_path = os.path.join(folder_path, file)
                
                # 輸出 txt 檔案路徑 - 正確的寫法
                # 取得檔案名稱（不含副檔名）
                file_name_without_ext = os.path.splitext(file)[0]
                file_name_without_ext = re.sub(r'_(MA|HE|EX|SE|OD)$', '', file_name_without_ext)

                output_txt_path = os.path.join(output_folder, f"{file_name_without_ext}.txt")
            
                
                # print(f"處理: {file} -> {output_txt_path}")
                
                # 讀取圖片
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"找不到圖片: {img_path}")
                    # 如果找不到圖片，仍然建立空的 txt 檔案
                    open(output_txt_path, 'w').close()
                    print(f"建立空的 txt 檔案: {output_txt_path}")
                else:
                    # 獲取圖片尺寸
                    nc, nr = img.shape[:2]  # nc = 寬度, nr = 高度
                    # print(f"圖片尺寸: {nc} x {nr}")
                    
                    # 轉換為 YOLO 格式並保存為 txt
                    Image_turn_to_yolo(img, nc, nr, output_txt_path, label)
            label += 1    

print("所有檔案處理完成！")