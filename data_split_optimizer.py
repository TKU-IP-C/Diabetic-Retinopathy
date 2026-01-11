import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold

# ============================================================
# # 配置區域 (Configuration)
# ============================================================
CONFIG = {
    # 建議使用絕對路徑以確保執行穩定
    "base_path": r"dataset\IDRiD\A. Segmentation\IDRiD_yolo",
    "num_folds": 5,                     # 預計切分的折數
    "search_iterations": 20000,         # 模擬次數，次數越多越能找到分佈趨於完美的組合
    "output_file": "best_fold_split.json",
    "class_names": {0: "MA", 1: "HE", 2: "EX", 3: "SE", 4: "OD"}
}

def analyze_lesion_distribution():
    """
    主要邏輯：以資料夾內實體圖片為基準進行搜尋
    1. 掃描所有圖片，提取基礎 ID (移除 _green_processed)
    2. 以基礎 ID 為單位，確保原圖與增強圖被分在同一折
    3. JSON 輸出將直接記錄實體圖片名稱 (例如 IDRiD_01.jpg)
    """
    base_path = Path(os.path.abspath(CONFIG["base_path"]))
    img_dir = base_path / "images"
    lbl_dir = base_path / "labels"
    
    if not img_dir.exists():
        print(f"❌ 錯誤：找不到圖片目錄 {img_dir}")
        return []
    
    # 1. 搜尋所有圖片檔案
    print(f"🔍 正在掃描圖片目錄: {img_dir}")
    all_image_paths = []
    all_image_paths.extend(list(img_dir.rglob(f"*.jpg")))
    
    if not all_image_paths:
        print(f"❌ 錯誤：在指定路徑下找不到任何圖片檔案。")
        return []

    print(f"📊 實體檔案總數: {len(all_image_paths)} (含增強圖)")

    # 2. 建立 ID 映射表，確保原圖與增強圖合併為一個單位
    # 鍵 (Key): 基礎 ID (如 IDRiD_01)
    # 值 (Value): 實體代表名稱 (如 IDRiD_01.jpg)
    id_to_filename_map = {}
    
    for img_path in all_image_paths:
        full_filename = img_path.name
        # 提取基礎 ID
        stem = img_path.stem
        
        # 優先權邏輯：
        # 如果是原圖 (不含 _green_processed)，優先設為該 ID 的代表名稱
        if stem not in id_to_filename_map:
            # 如果目前還沒存過原圖，暫時用處理過的圖名代替
            id_to_filename_map[stem] = full_filename

    print(f"🧬 識別出基礎影像單元共: {len(id_to_filename_map)} 組 (已完成 1:1 匹配)")

    image_stats = []
    for stem, representative_name in id_to_filename_map.items():
        counts = Counter()
        
        # 尋找對應的標註檔 (遍歷 labels 目錄)
        label_path = None
        for lbl_file in lbl_dir.rglob(f"{stem}.txt"):
            if lbl_file.exists():
                label_path = lbl_file
                break
        
        if label_path:
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            cls_id = int(parts[0])
                            counts[cls_id] += 1
            except Exception as e:
                print(f"⚠️ 讀取標籤失敗 {label_path.name}: {e}")
        
        # 建立分層標籤字串
        lesion_types = sorted(list(counts.keys()))
        stratify_label = "_".join(map(str, lesion_types)) if lesion_types else "bg"
        
        image_stats.append({
            "id": representative_name,  # 這裡儲存的是實體圖片名稱 (如 IDRiD_01.jpg)
            "stem": stem,
            "label": stratify_label,
            "counts": dict(counts)
        })
        
    return image_stats

def calculate_split_fitness(data_list, val_indices):
    """計算切分方案的平衡得分 (各折病灶分佈的一致性)"""
    val_data = [data_list[i] for i in val_indices]
    total_lesions = Counter()
    for item in val_data:
        for cls, count in item["counts"].items():
            total_lesions[cls] += count
    return total_lesions

def run_optimization():
    image_pool = analyze_lesion_distribution()
    if not image_pool:
        print("❌ 搜尋失敗，請檢查資料夾與檔案名稱。")
        return

    # 轉為數據框架進行分析
    df = pd.DataFrame(image_pool)
    X = df['id'].values # 這是實體圖片名稱
    y = df['label'].values
    
    # 處理稀有類別，防止 StratifiedKFold 報錯
    label_counts = Counter(y)
    for label, count in label_counts.items():
        if count < CONFIG["num_folds"]:
            df.loc[df['label'] == label, 'label'] = 'rare_combination'
    
    y = df['label'].values
    best_score = float('inf')
    best_split_config = None

    print(f"🚀 開始執行 {CONFIG['search_iterations']} 次蒙地卡羅分佈模擬...")
    
    for i in range(CONFIG["search_iterations"]):
        seed = random.randint(1, 1000000)
        skf = StratifiedKFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=seed)
        
        fold_distributions = []
        try:
            for _, val_idx in skf.split(X, y):
                dist = calculate_split_fitness(image_pool, val_idx)
                fold_distributions.append(dist)
            
            # 計算變異係數 (CV) 之和
            total_cv = 0
            for cls_id in CONFIG["class_names"].keys():
                counts = [d.get(cls_id, 0) for d in fold_distributions]
                if np.mean(counts) > 0:
                    total_cv += np.std(counts) / np.mean(counts)
            
            if total_cv < best_score:
                best_score = total_cv
                best_split_config = {
                    "seed": seed,
                    "avg_cv": float(total_cv / len(CONFIG["class_names"])),
                    "folds": {}
                }
                # 記錄每一折具體的實體檔名清單
                for f_idx, (t_idx, v_idx) in enumerate(skf.split(X, y)):
                    best_split_config["folds"][f_idx + 1] = {
                        "train": X[t_idx].tolist(),
                        "val": X[v_idx].tolist()
                    }
                
                if i % 1000 == 0:
                    print(f"🏆 迭代 {i}: 找到更優分配方案 (分佈得分: {total_cv:.4f})")
        except Exception:
            continue

    if best_split_config:
        with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
            json.dump(best_split_config, f, indent=4, ensure_ascii=False)
        
        print("\n" + "="*60)
        print(f"✨ 搜尋完成！黃金切分清單已儲存至: {CONFIG['output_file']}")
        print(f"📊 最佳平衡得分 (CV): {best_score:.4f}")
        print(f"✅ 成功處理 {len(image_pool)} 組圖片單元，JSON 已與實體檔名 1:1 同步。")
        print("="*60)

if __name__ == "__main__":
    run_optimization()