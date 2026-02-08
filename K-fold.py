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
    "model_variant": "yolov12n.pt",    # 使用 YAML 配置文件
    
    # --- 2. 核心訓練控制 (Core Training Control) ---
    "epochs": 150,                       # 總訓練輪數
    "imgsz": 1080,                       # 輸入影像解析度
    "batch": 8,                         # 批次大小
    "patience": 50,                     # 早停機制耐心值
    "device": 0,
    "workers": 0,                       # 資料讀取執行緒數
    "exist_ok": True,                   # 是否允許覆蓋已存在的實驗資料夾
    "rect": True,                       # 是否啟用矩形訓練以加速

    # --- 3. 優化器與學習率 (Optimizer & Learning Rate) ---
    # "lr0": 0.01,                       # 初始學習率
    # "lrf": 0.01,                       # 最終學習率比例 (最終學習率 = lr0 * lrf)
    # "momentum": 0.937,                 # 優化器動量
    # "weight_decay": 0.0005,            # 權重衰減（L2 正則化），用於防止模型過度擬合
    # "optimizer": "SGD",                # 優化器類型 (auto 會根據環境選用 AdamW 或 SGD)
    
    # --- 4. 損失函數權重 (Loss Weights / Gains) ---
    # "box": 7.5,                        # 邊界框位置損失權重
    # "cls": 0.5,                        # 分類損失權重
    # "dfl": 1.5,                        # 分佈焦點損失權重
    
    # --- 5. 數據增強 (Data Augmentation) ---
    # "mosaic": 1.0,                     # 馬賽克增強比例
    # "mixup": 0.0,                      # MixUp 增強比例
    # "copy_paste": 0.0,                 # 複製貼上增強機率
    # "fliplr": 0.5,                     # 左右翻轉影像的機率
    
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
    """取得所有影像檔的基礎 ID（移除增強後綴）"""
    image_pool = []
    img_dir = Path(TRAINING_CONFIG["base_path"]) / "images"
    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    
    # 遍歷所有影像
    for ext in valid_exts:
        for img_path in img_dir.rglob(f"*{ext}"):
            # 取得基礎 ID（移除增強後綴如 '_aug' 等）
            base_stem = img_path.stem
            # 移除常見的增強後綴
            for suffix in ['_aug', '_flip', '_rotate', '_crop']:
                if base_stem.endswith(suffix):
                    base_stem = base_stem[:-len(suffix)]
                    break
            
            image_pool.append((base_stem, img_path.suffix))
    
    # 移除重複的基礎 ID
    return sorted(list(set(image_pool)))

def create_fold_yaml(exp_dir, fold_idx, train_ids, val_ids):
    """為每一折建立專屬 YAML 配置檔與路徑清單"""
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    train_txt = configs_dir / f"train_f{fold_idx}.txt"
    val_txt = configs_dir / f"val_f{fold_idx}.txt"
    
    img_dir = Path(TRAINING_CONFIG["base_path"]) / "images"

    def write_paths(file_path, id_list):
        """將圖片路徑寫入 txt 文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for img_id, ext in id_list:
                # 優先檢查 images 根目錄下的文件
                origin = img_dir / f"{img_id}{ext}"
                if origin.exists():
                    f.write(str(origin.absolute()).replace('\\', '/') + '\n')
                else:
                    # 如果根目錄沒有，檢查 train 或 val 子目錄
                    for split in ['train', 'val']:
                        split_path = img_dir / split / f"{img_id}{ext}"
                        if split_path.exists():
                            f.write(str(split_path.absolute()).replace('\\', '/') + '\n')
                            break

    write_paths(train_txt, train_ids)
    write_paths(val_txt, val_ids)
    
    # 建立 YAML 配置文件
    yaml_content = f"""path: {str(Path(TRAINING_CONFIG["base_path"]).absolute()).replace('\\', '/')}
train: {str(train_txt.absolute()).replace('\\', '/')}
val: {str(val_txt.absolute()).replace('\\', '/')}
nc: {len(TRAINING_CONFIG['names'])}
names: {TRAINING_CONFIG['names']}
"""
    
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
        print("❌ 錯誤: 找不到圖片資料，請檢查路徑設定。")
        return
    
    print(f"📊 總共找到 {len(base_pool)} 張唯一影像")

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
        print(f"  訓練集: {len(train_idx)} 張, 驗證集: {len(val_idx)} 張")
        
        train_ids = [base_pool[idx] for idx in train_idx]
        val_ids = [base_pool[idx] for idx in val_idx]
        
        yaml_path = create_fold_yaml(experiment_dir, fold_num, train_ids, val_ids)
        
        # 載入模型 - 使用 YAML 配置文件
        print(f"  載入模型: {TRAINING_CONFIG['model_variant']}")
        model = YOLO(TRAINING_CONFIG["model_variant"])

        # 組合訓練參數
        train_args = {
            "data": str(yaml_path),
            "project": str(experiment_dir),
            "name": f"fold_{fold_num}",
            "epochs": TRAINING_CONFIG["epochs"],
            "imgsz": TRAINING_CONFIG["imgsz"],
            "batch": TRAINING_CONFIG["batch"],
            "patience": TRAINING_CONFIG["patience"],
            "device": TRAINING_CONFIG["device"],
            "workers": TRAINING_CONFIG["workers"],
            "exist_ok": TRAINING_CONFIG["exist_ok"],
            "rect": TRAINING_CONFIG["rect"],
            "save": True,
            "verbose": True,
            "pretrained": True,  # 使用預訓練權重
        }
        
        # 添加可選參數（如果存在於配置中）
        optional_args = ["lr0", "lrf", "momentum", "weight_decay", "optimizer",
                        "box", "cls", "dfl", "mosaic", "mixup", "copy_paste", "fliplr"]
        
        for arg in optional_args:
            if arg in TRAINING_CONFIG:
                train_args[arg] = TRAINING_CONFIG[arg]

        print(f"  開始訓練第 {fold_num} 折...")
        
        # 開始訓練此折
        results = model.train(**train_args)
        
        # 驗證該折效能
        print(f"  驗證第 {fold_num} 折效能...")
        val_results = model.val()
        
        results_map50.append(val_results.box.map50)
        print(f"✅ 第 {fold_num} 折完畢，mAP50: {val_results.box.map50:.4f}")

        # 清理顯存
        del model 
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 生成統計摘要報告
    summary_path = experiment_dir / "summary_report.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment ID: {experiment_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {TRAINING_CONFIG['model_variant']}\n")
        f.write(f"Total Folds: {len(kf_splits)}\n")
        f.write(f"Total Images: {len(base_pool)}\n")
        f.write("-" * 50 + "\n")
        f.write("Fold Results (mAP50):\n")
        for idx, val in enumerate(results_map50):
            f.write(f"  Fold {idx+1}: {val:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final Statistics:\n")
        f.write(f"  Average mAP50: {np.mean(results_map50):.4f}\n")
        f.write(f"  Std Deviation: {np.std(results_map50):.4f}\n")
        f.write(f"  Min mAP50: {np.min(results_map50):.4f}\n")
        f.write(f"  Max mAP50: {np.max(results_map50):.4f}\n")

    print(f"\n" + "=" * 60)
    print(f"✅ 實驗總結完成！")
    print(f"📊 平均 mAP50: {np.mean(results_map50):.4f} (±{np.std(results_map50):.4f})")
    print(f"📁 摘要報告已存至: {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_kfold_training()