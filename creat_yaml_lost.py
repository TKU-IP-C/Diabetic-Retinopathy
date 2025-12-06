import os
from pathlib import Path
def create_yaml_config():
    """建立 YOLO 資料集配置文件"""
    
    yaml_content = f"""# YOLO 糖尿病視網膜病變資料集配置
path: {os.path.abspath(r"dataset\IDRiD\A. Segmentation\IDRiD_yolo")}  # 資料集根目錄
train: {os.path.abspath(r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\images\train")}   # 訓練圖片路徑
val: {os.path.abspath(r"dataset\IDRiD\A. Segmentation\IDRiD_yolo\images\val ")}      # 驗證圖片路徑

# 類別名稱 (根據您的病灶類型修改)
nc: 4  # 類別數量
names:
    0: Microaneurysms  # 微動脈瘤
    1: Haemorrhages  # 出血
    2: Hard Exudates + Soft Exudates  #硬性滲出物 + 軟性滲出物
    3: Optic Disc  # 視神經盤

# 下載指令/說明
# download: |
#   # 在這裡添加下載指令或說明
"""
    
    yaml_path = r"diabetic_retinopathy.yaml"
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ YAML 配置文件已建立: {yaml_path}")
    return yaml_path

# 建立配置文件
yaml_path = create_yaml_config()