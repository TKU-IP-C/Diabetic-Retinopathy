import cv2
import os
import numpy as np

def process_images_with_opencv(input_folder, output_folder, kernel_size=(5, 5), sigma=0):
    """使用OpenCV处理图片，只保留绿色通道并添加高斯模糊"""
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            # 读取图片
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # 方法1: 先提取绿色通道再模糊
                # 创建全零矩阵（黑色背景）
                green_channel = np.zeros_like(img)
                
                # 只保留绿色通道（在OpenCV中是BGR格式，所以索引1是绿色）
                green_channel[:, :, 1] = img[:, :, 1]
                
                # 对绿色通道应用高斯模糊
                green_blurred = cv2.GaussianBlur(green_channel, kernel_size, sigma)
                
                # 保存处理后的图片
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, green_blurred)
                print(f"已处理: {filename}")
            else:
                print(f"无法读取: {filename}")


# 使用示例
input_folder = r"dataset\IDRiD\A. Segmentation\1. Original Images\b. Testing Set"  # 输入文件夹路径
output_folder = r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\images\val"  # 输出文件夹路径
process_images_with_opencv(input_folder, output_folder)