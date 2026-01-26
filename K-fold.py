import torch
import numpy as np
import gc
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from ultralytics import YOLO

# ============================================================
# # 訓練參數 
# ============================================================
TRAINING_CONFIG = {
    # --- 1. 實驗與路徑設定 (Experiment & Path) ---
    "k_fold": 5,                        # 交叉驗證折數
    "base_path": r"dataset\IDRiD\A. Segmentation\IDRiD_yolo", # 資料集根目錄路徑
    "results_root": "runs/kfold_experiments", # 所有實驗結果儲存的根目錄名稱
    "model_variant": "yolov12n.pt",     # 模型版本
    
    # --- 2. 核心訓練控制 (Core Training Control) ---
    "epochs": 150,                       # 總訓練輪數
    "imgsz": 1080,                       # 輸入影像解析度
    "batch": 8,                         # 批次大小
    "patience": 50,                     # 早停機制耐心值
    "device": 0,
    "workers": 0,                       # 資料讀取執行緒數
    "exist_ok": True,                   # 是否允許覆蓋已存在的實驗資料夾
    "rect": True,                       # 是否啟用矩形訓練以加速

    # # --- 3. 優化器與學習率 (Optimizer & Learning Rate) ---
    # "lr0": 0.005,                       # 初始學習率
    # "lrf": 0.01,                        # 最終學習率比例 (最終學習率 = lr0 * lrf)
    "momentum": 0.8,                      # 優化器動量
    # "weight_decay": 0.0005,             # 權重衰減（L2 正則化），用於防止模型過度擬合
    "optimizer": "SGD",                   # 優化器類型 (auto 會根據環境選用 AdamW 或 SGD)
    "iou": 0.5,                           # IoU 閾值，用於決定正負樣本
    
    # # --- 4. 損失函數權重 (Loss Weights / Gains) ---
    "box": 16.5,                         # 邊界框位置損失權重，數值高則更要求定位精準度
    "cls": 1.5,                          # 分類損失權重，衡量辨識病灶種類的準確率
    # "dfl": 3.0,                        # 分佈焦點損失權重，細化邊界框的邊緣學習
    
    # # --- 5. 數據增強 (Data Augmentation) ---
    # "dropout": 0.4,                   # 隨機關閉神經元比例，增加模型的泛化能力
    "mosaic": 1.0,                      # 馬賽克增強比例 (1.0 = 100%)，將四張圖拼成一張，對小物件辨識極有幫助
    # "copy_paste": 0,                  # 複製貼上增強機率，將物件實例隨機複製到其他圖上
    # "fliplr": 0,                      # 左右翻轉影像的機率 (50%)
    
    # --- 6. 類別定義 (Class Definitions) ---
    "names": {
        0: "MA",    # 微動脈瘤
        1: "HE",    # 出血點
        2: "EX",    # 硬性
        3: "SE",    # 軟性滲出物
        4: "OD"     # 視盤
    }
}

# ============================================================
# --- 工具與處理函式 (Utility Functions) ---
# ============================================================

def get_base_id_pool():
    
    image_pool = []
    img_dir = Path(TRAINING_CONFIG["base_path"]) / "images"
    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    
    for split in ['train', 'val']:
        split_dir = img_dir / split
        if not split_dir.exists(): continue
        for ext in valid_exts:
            for img_path in split_dir.glob(f"*{ext}"):
                # 取得基礎 ID（移除增強後綴）
                # 範例: IDRiD_01_green_processed -> IDRiD_01
                base_stem = img_path.stem
                image_pool.append((base_stem, img_path.suffix))
                
    return sorted(list(set(image_pool)))

def create_fold_yaml(exp_dir, fold_idx, train_ids, val_ids):
    """為每一折建立專屬 YAML 配置檔與路徑清單"""
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    train_txt = configs_dir / f"train_f{fold_idx}.txt"
    val_txt = configs_dir / f"val_f{fold_idx}.txt"
    
    img_dir = Path(TRAINING_CONFIG["base_path"]) / "images"

    def write_paths(file_path, id_list):
        """將圖片絕對路徑寫入 txt，支援單獨原圖或單獨綠色通道圖"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for img_id, ext in id_list:
                for split in ['train', 'val']:
                    origin = img_dir / split / f"{img_id}{ext}"
                    green = img_dir / split / f"{img_id}_green_processed{ext}"
                    
                    # 只要檔案存在就寫入路徑，兩者互不依賴
                    if origin.exists():
                        f.write(str(origin.absolute()).replace('\\', '/') + '\n')
                    if green.exists():
                        f.write(str(green.absolute()).replace('\\', '/') + '\n')

    write_paths(train_txt, train_ids)
    write_paths(val_txt, val_ids)
    
    # 處理路徑字串以建立 YAML
    base_abs = str(Path(TRAINING_CONFIG["base_path"]).absolute()).replace('\\', '/')
    train_abs = str(train_txt.absolute()).replace('\\', '/')
    val_abs = str(val_txt.absolute()).replace('\\', '/')
    
    yaml_content = f"path: {base_abs}\ntrain: {train_abs}\nval: {val_abs}\nnc: {len(TRAINING_CONFIG['names'])}\nnames:\n"
    for idx, name in TRAINING_CONFIG["names"].items():
        yaml_content += f"  {idx}: {name}\n"
        
    yaml_path = configs_dir / f"fold_{fold_idx}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    return yaml_path

def run_kfold_training():
    """執行 K-Fold 交叉驗證訓練流程"""
    # 建立本次實驗的唯一資料夾 (以時間命名)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"KFold_Run_{timestamp}"
    experiment_dir = Path(TRAINING_CONFIG["results_root"]) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 啟動實驗: {experiment_name}")
    print(f"📂 儲存路徑: {experiment_dir.absolute()}")
    print("=" * 60)

    base_pool = get_base_id_pool()
    if not base_pool:
        print("❌ 錯誤: 找不到圖片資料，請檢查路徑設定（必須包含 images/train 或 images/val）。")
        return

    # 處理折數切分
    k_val = TRAINING_CONFIG["k_fold"]
    if k_val < 2:
        print("💡 檢測到 K=1，將執行標準 80/20 訓練模式。")
        split_point = int(len(base_pool) * 0.8)
        kf_splits = [(np.arange(split_point), np.arange(split_point, len(base_pool)))]
    else:
        kf = KFold(n_splits=k_val, shuffle=True, random_state=42)
        kf_splits = list(kf.split(base_pool))
    
    results_map50 = []

    # 遍歷每一折進行訓練
    for i, (train_idx, val_idx) in enumerate(kf_splits):
        fold_num = i + 1
        print(f"\n🌀 [折數 {fold_num} / {len(kf_splits)}] 訓練準備中...")
        
        train_ids = [base_pool[idx] for idx in train_idx]
        val_ids = [base_pool[idx] for idx in val_idx]
        
        yaml_path = create_fold_yaml(experiment_dir, fold_num, train_ids, val_ids)
        
        # 每一折都從基礎預訓練權重開始
        model = YOLO(TRAINING_CONFIG["model_variant"])

        # 組合訓練參數
        train_args = {
            "data": str(yaml_path),
            "project": str(experiment_dir),
            "name": f"fold_{fold_num}",
            "save": True,
            "verbose": True,
        }
        
        # 動態載入 TRAINING_CONFIG 中的其餘參數
        exclude_keys = ["k_fold", "base_path", "results_root", "model_variant", "names"]
        for key, value in TRAINING_CONFIG.items():
            if key not in exclude_keys:
                train_args[key] = value

        # 開始訓練此折
        metrics = model.train(**train_args)
        
        # 驗證該折效能並記錄
        # metrics = model.val()
        results_map50.append(metrics.box.map50)
        print(f"✅ 第 {fold_num} 折完畢，當前 mAP50: {metrics.box.map50:.4f}")

        # --- 核心修正：強制清理顯存，放在每一折結束時 ---
        del model 
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        # ---------------------------------------------

    # 4. 生成統計摘要報告
    summary_path = experiment_dir / "summary_report.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment ID: {experiment_name}\n")
        f.write(f"Total Folds: {len(kf_splits)}\n")
        f.write("-" * 40 + "\n")
        for idx, val in enumerate(results_map50):
            f.write(f"Fold {idx+1}: mAP50 = {val:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Statistics:\n")
        f.write(f" > Average mAP50: {np.mean(results_map50):.4f}\n")
        f.write(f" > Std Deviation: {np.std(results_map50):.4f}\n")

    print(f"\n" + "=" * 60)
    print(f"✅ 實驗總結完成！")
    print(f"📊 平均 mAP50: {np.mean(results_map50):.4f} (±{np.std(results_map50):.4f})")
    print(f"📁 摘要報告已存至: {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_kfold_training()