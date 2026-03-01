import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import threading
import queue
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
from ultralytics import YOLO

# ====================== 模型載入部分（請你替換成自己的） ======================

# 假設你的模型類別（如果你用 .pth 通常需要這個）
class YourDRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ... 你的模型結構，例如 ResNet / EfficientNet ...
        self.fc = torch.nn.Linear(512, 5)  # 假設 5 級分類

    def forward(self, x):
        # ... 
        return x

# 全局變數或在類別裡存
model_pth = None
model_pt  = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DRResNet152(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.resnet152(weights=None)
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
def load_models():
    global model_pth, model_pt
    
    model_pth = DRResNet152().to(device)  # 注意：沒有 num_classes 參數了

    state_dict = torch.load("./best_model.pth", map_location=device)
    
    # 先嘗試嚴格載入，如果失敗再用 strict=False
    try:
        model_pth.load_state_dict(state_dict)
    except RuntimeError as e:
        print("嚴格載入失敗，嘗試忽略不匹配的 key：", e)
        model_pth.load_state_dict(state_dict, strict=False)
    
    model_pth.eval()

    model_pt = YOLO("./best.pt")

# 預處理（常見 fundus 圖片前處理，請依你的模型調整)
transform = transforms.Compose([
    transforms.Resize((512, 512)),          # 改成你模型輸入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

severity_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def predict_pth(image_pil):
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_pth(img_tensor)          # shape: [1, 1]
        score = output.item()                   # 取出 scalar 值

    severity_score = max(0.0, min(4.0, score))  # 夾在 0~4 之間
    severity_text = f"嚴重程度分數：{severity_score:.2f} / 4.0"

    # 信心度可以簡單用距離邊界的程度估計，或直接省略
    confidence = 100.0 - abs(severity_score - 2.0) * 25.0  # 範例：越接近中間越不確定，可自訂

    return severity_text, severity_score, confidence

def predict_pt(image_pil):
    # PIL → numpy BGR
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # CLAHE 前處理（如果你確定訓練時有做；YOLO 通常不需要，但保留可選）
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 建議 tileGridSize 更大一點
    enhanced = clahe.apply(gray)
    img_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)   # 轉回 3 通道
    
    # YOLO 推理（直接傳 BGR numpy array）
    results = model_pt(img_enhanced)   # 或直接傳 img_cv 如果不做 CLAHE

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = model_pt.names[cls_id]
            conf = float(box.conf)
            xyxy = [round(float(x)) for x in box.xyxy[0]]  # 轉整數比較好看
            detections.append(f"{label} 信心 {conf:.2f} 位置 {xyxy}")

    if not detections:
        return "無偵測到任何病灶", "N/A", 0.0
    
    return "\n".join(detections)

# ====================== UI 部分 ======================

class DRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("糖尿病視網膜病變 - 雙模型分析")
        self.root.geometry("1080x960")

        self.load_models_thread = threading.Thread(target=load_models, daemon=True)
        self.load_models_thread.start()

        # 上方 - 圖片顯示區
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        # 選擇檔案按鈕
        ttk.Button(root, text="選擇眼底照片", command=self.open_image).pack(pady=5)

        # 分析按鈕
        self.btn_analyze = ttk.Button(root, text="開始分析", command=self.start_analysis, state="disabled")
        self.btn_analyze.pack(pady=10)

        # 結果顯示區（scrolled text）
        self.result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 12), height=15)
        self.result_text.pack(padx=20, pady=10, fill="both", expand=True)

        self.queue = queue.Queue()
        self.current_image = None
        self.root.after(300, self.check_queue)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.current_image = Image.open(path).convert("RGB")
            # 顯示縮圖
            img_show = self.current_image.resize((400, 300))
            photo = ImageTk.PhotoImage(img_show)
            self.img_label.config(image=photo)
            self.img_label.image = photo  # 保持參考
            self.btn_analyze.config(state="normal")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "已載入圖片，點擊「開始分析」\n")

    def start_analysis(self):
        if not self.current_image:
            return
        self.btn_analyze.config(state="disabled")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "分析中...\n\n")

        threading.Thread(target=self.run_analysis, daemon=True).start()

    def run_analysis(self):
        try:
            # .pth 回歸模型
            self.queue.put("【.pth 模型】嚴重程度分析中...\n")

            severity_text, score, conf = predict_pth(self.current_image)

            msg1 = (
                f"{severity_text}\n"
                f"估計信心度：{conf:.1f}%\n\n"
            )
            self.queue.put(msg1)

            # .pt YOLO 偵測模型
            self.queue.put("【.pt 模型】病灶偵測中...\n")

            detection_result = predict_pt(self.current_image)

            msg2 = f"{detection_result}\n"
            self.queue.put(msg2)

        except Exception as e:
            self.queue.put(f"錯誤：{str(e)}\n")

        finally:
            self.queue.put("DONE")

    def check_queue(self):
        try:
            while True:
                line = self.queue.get_nowait()
                if line == "DONE":
                    self.btn_analyze.config(state="normal")
                else:
                    self.result_text.insert(tk.END, line)
                    self.result_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(200, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = DRApp(root)
    root.mainloop()