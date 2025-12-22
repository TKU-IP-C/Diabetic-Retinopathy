from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
from pathlib import Path

def train_diabetic_retinopathy_model():
    """訓練糖尿病視網膜病變檢測模型"""
    
    print("\n🎯 開始訓練糖尿病視網膜病變檢測模型...")
    
    yaml_path = r"diabetic_retinopathy.yaml"
    
    try:
        # 加載模型
        print("\n📦 加載YOLO模型...")
        model = YOLO('yolov12n.pt')
        
        # 訓練參數
        train_args = {
            'data': yaml_path,
            'epochs': 70,
            'imgsz': 1280,        # 使用較高解析度的圖片
            'batch': 8,           # 視網膜圖片較大，使用較小的批次
            'patience': 20,       # 早停耐心值
            'save': True,
            'device': '0',
            'project': 'runs/detect',
            'name': 'DR_sz1280_train70_green_v5_yolov12n_batch8',
            'exist_ok': True,     # 允許覆蓋現有實驗
            'verbose': True,      # 顯示詳細輸出
            'augment': True,      # 使用資料增強
            'lr0': 0.01,          # 初始學習率
            'lrf': 0.2,           # 最終學習率
            'box': 1.5,           # 邊界框損失增
            'hsv_h': 0.015,   # 色調變化 (default 0.015)
            'hsv_s': 0.7,     # 飽和度變化
            'hsv_v': 0.4,     # 亮度變化
            # 'degrees': 10.0,  # 旋轉 ±10度
            # 'translate': 0.2, # 平移 ±20%
            'scale': 0.9,     # 縮放 (可加大到 0.9~1.0 幫助小目標)
            'shear': 10.0,    # 剪切
            # 'perspective': 0.0001,  # 透視變換
            # 'flipud': 0.5,    # 上下翻轉 50%
            # 'fliplr': 0.5,    # 左右翻轉 50%
            'mosaic': 1.0,    # Mosaic augmentation (強烈推薦開啟，對小資料集超有效)
            'mixup': 0.3,     # MixUp (可試 0.3~0.5)
            # 'freeze': 10,  # 凍結前 10 層
        }
        
        print("🚀 開始訓練...")
        print("這可能需要一些時間，請耐心等待...")
        
        # 開始訓練
        results = model.train(**train_args)
        
        print("✅ 訓練完成！")
        
        # 顯示訓練結果
        if hasattr(results, 'results_dict'):
            print("\n📈 訓練結果:")
            for key, value in results.results_dict.items():
                print(f"  {key}: {value:.4f}")
        
        return model, results
        
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        return None, None

def validate_model(model):
    """驗證訓練好的模型"""
    if model is None:
        return
    
    print("\n🔍 驗證模型...")
    
    try:
        # 在驗證集上評估模型
        metrics = model.val()
        
        print("✅ 驗證完成！")
        print(f"📊 mAP50: {metrics.box.map50:.4f}")
        print(f"📊 mAP50-95: {metrics.box.map:.4f}")
        print(f"📊 精確度: {metrics.box.precision:.4f}")
        print(f"📊 召回率: {metrics.box.recall:.4f}")
        
    except Exception as e:
        print(f"❌ 驗證過程中發生錯誤: {e}")

def main():
    """主函數"""
    print("=" * 60)
    print("🩺 糖尿病視網膜病變檢測模型訓練")
    print("=" * 60)

    # 訓練模型
    model, results = train_diabetic_retinopathy_model()
    
    # 驗證模型
    if model:
        validate_model(model)
        
        print("\n🎉 訓練流程完成！")
        print("📁 訓練結果保存在: runs/detect/")
        print("💡 您可以使用訓練好的模型進行預測:")
        print("   results = model('path/to/image.jpg')")
    
    print("\n" + "=" * 60)

# 執行主函數
if __name__ == "__main__":
    main()