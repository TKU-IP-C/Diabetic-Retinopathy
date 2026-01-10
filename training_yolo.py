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
            'imgsz': 800,        # 使用較高解析度的圖片
            'batch': 8,           # 視網膜圖片較大，使用較小的批次
            'patience': 20,       # 早停耐心值
            'save': True,
            'device': '0',
            'project': 'runs/detect',
            'name': 'DR_sz800_train70_v1',
            'exist_ok': True,     # 允許覆蓋現有實驗
            'verbose': True,      # 顯示詳細輸出
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