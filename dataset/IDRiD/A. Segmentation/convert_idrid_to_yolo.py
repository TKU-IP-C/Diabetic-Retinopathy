import os
import cv2
import numpy as np
from PIL import Image
import shutil

def convert_mask_to_yolo_bbox(mask_path, class_id, img_width, img_height):
    """
    將分割遮罩轉換為YOLO格式的邊界框
    """
    # 讀取遮罩圖像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    # 找到輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # 過濾太小的區域
            x, y, w, h = cv2.boundingRect(contour)
            
            # 轉換為YOLO格式 (歸一化座標)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            
            bboxes.append([class_id, x_center, y_center, width, height])
    
    return bboxes

def process_idrid_dataset(root_path, output_path):
    """
    處理整個IDRiD資料集
    """
    # 類別映射
    class_mapping = {
        'MA': 0,  # 微動脈瘤
        'HE': 1,  # 出血
        'EX': 2,  # 硬性滲出物
        'SE': 3   # 軟性滲出物/棉絮斑
    }
    
    # 創建輸出目錄
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
    
    # 處理訓練集和測試集
    for split in ['Training', 'Testing']:
        split_key = 'train' if split == 'Training' else 'val'
        
        # 原始圖像路徑
        images_dir = os.path.join(root_path, 'A. Segmentation', '1. Original Images', 
                                 'a. Training Set' if split == 'Training' else 'b. Testing Set')
        
        # 遮罩路徑
        masks_dir = os.path.join(root_path, 'A. Segmentation', '2. All Segmentation Groundtruths',
                               'a. Training Set' if split == 'Training' else 'b. Testing Set')
        
        # 獲取所有圖像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext.lower().replace('*', ''))])
        
        print(f"處理 {split} 集，共 {len(image_files)} 張圖片")
        
        for img_file in image_files:
            # 圖像基礎名稱 (不含擴展名)
            base_name = os.path.splitext(img_file)[0]
            
            # 讀取圖像獲取尺寸
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"無法讀取圖像: {img_path}")
                continue
                
            img_height, img_width = img.shape[:2]
            
            all_bboxes = []
            
            # 處理每個類別的遮罩
            for class_name, class_id in class_mapping.items():
                class_mask_dir = os.path.join(masks_dir, class_name)
                
                # 尋找對應的遮罩文件
                mask_files = [f for f in os.listdir(class_mask_dir) if f.startswith(base_name)]
                
                for mask_file in mask_files:
                    mask_path = os.path.join(class_mask_dir, mask_file)
                    bboxes = convert_mask_to_yolo_bbox(mask_path, class_id, img_width, img_height)
                    all_bboxes.extend(bboxes)
            
            # 如果有檢測到物件，保存標籤文件
            if all_bboxes:
                # 保存標籤文件
                label_filename = base_name + '.txt'
                label_path = os.path.join(output_path, 'labels', split_key, label_filename)
                
                with open(label_path, 'w') as f:
                    for bbox in all_bboxes:
                        line = ' '.join([str(x) for x in bbox])
                        f.write(line + '\n')
                
                # 複製圖像文件到輸出目錄
                output_img_path = os.path.join(output_path, 'images', split_key, base_name + '.jpg')
                
                # 如果是tif文件，轉換為jpg
                if img_file.lower().endswith(('.tif', '.tiff')):
                    # 使用PIL讀取並保存為JPG
                    pil_img = Image.open(img_path)
                    # 轉換為RGB模式（如果需要的話）
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    pil_img.save(output_img_path, 'JPEG', quality=95)
                else:
                    # 對於其他格式，直接複製
                    shutil.copy2(img_path, output_img_path)
                
                print(f"處理完成: {base_name}, 找到 {len(all_bboxes)} 個物件")
            else:
                print(f"未找到物件: {base_name}")
    
    print("資料集轉換完成！")
    
    # 創建data.yaml文件
    create_yaml_file(output_path, class_mapping)

def create_yaml_file(output_path, class_mapping):
    """
    創建YOLO需要的data.yaml配置文件
    """
    yaml_content = f"""# IDRiD 資料集配置文件
path: {os.path.abspath(output_path)}  # 資料集根目錄
train: images/train  # 訓練圖片目錄
val: images/val      # 驗證圖片目錄

# 類別名稱
names:
"""
    for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
        yaml_content += f"  {class_id}: {class_name}\n"
    
    yaml_file_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"配置文件已創建: {yaml_file_path}")

def check_dataset_stats(output_path):
    """
    檢查轉換後的資料集統計信息
    """
    print("\n=== 資料集統計 ===")
    
    for split in ['train', 'val']:
        labels_dir = os.path.join(output_path, 'labels', split)
        images_dir = os.path.join(output_path, 'images', split)
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        print(f"\n{split} 集:")
        print(f"  標籤文件數: {len(label_files)}")
        print(f"  圖像文件數: {len(image_files)}")
        
        # 統計每個類別的物件數量
        class_counts = {0:0, 1:0, 2:0, 3:0}
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
        
        print(f"  物件數量統計:")
        class_names = {0: 'MA', 1: 'HE', 2: 'EX', 3: 'SE'}
        for class_id, count in class_counts.items():
            print(f"    {class_names[class_id]}: {count}")

if __name__ == "__main__":
    # 設置路徑
    idrid_root = "IDRiD"  # IDRiD資料集根目錄
    output_dir = "IDRiD_yolo"  # 輸出目錄
    
    # 執行轉換
    process_idrid_dataset(idrid_root, output_dir)
    
    # 檢查統計信息
    check_dataset_stats(output_dir)