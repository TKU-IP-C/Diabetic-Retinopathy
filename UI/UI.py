import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import threading
import queue
import torch
from torchvision import transforms

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

def load_models():
    global model_pth, model_pt
    model_pt = YOLO("best.pt")

    # 先載 .pth （假設是 state_dict）
    model_pth = YourDRModel().to(device)               # 先建模型結構
    state_dict = torch.load("path/to/your_model.pth", map_location=device)
    model_pth.load_state_dict(state_dict)
    model_pth.eval()

    # 再載 .pt （假設是整個 model 或 scripted）
    model_pt = torch.load("path/to/your_model.pt", map_location=device)
    model_pt.eval()   # 如果是 nn.Module 就 eval()
    # 如果是 torch.jit.ScriptModule，就不需要再 .eval()

# 預處理（常見 fundus 圖片前處理，請依你的模型調整）
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # 改成你模型輸入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

severity_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def predict_pth(image_pil):
    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_pth(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
    return pred_class, confidence

def predict_pt(image_pil):
    # image_pil 是 PIL Image，從你的 UI 來的

    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)  # PIL → OpenCV BGR

    gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
    img_cv = clahe.apply(gray)

    results = model_pt(img_cv)  # model_pt 是 global 的 YOLO 物件

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            label = model_pt.names[cls]
            conf = box.conf.item()
            xyxy = box.xyxy[0].tolist()
            detections.append(f"{label} ({conf:.2f}): {xyxy}")

    # 回傳你要顯示的文字，或整個 annotated image
    return "\n".join(detections) if detections else "無偵測到異常"

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
            # 先跑 .pth
            self.queue.put("【.pth 模型】嚴重程度分析中...\n")
            class_idx, conf = predict_pth(self.current_image)
            msg1 = f"嚴重程度：{severity_labels[class_idx]} (等級 {class_idx})\n信心度：{conf:.4f}\n\n"
            self.queue.put(msg1)

            # 再跑 .pt
            self.queue.put("【.pt 模型】執行中...\n")
            # 依你的 .pt 模型實際輸出調整這裡
            class_idx_pt, conf_pt = predict_pt(self.current_image)
            msg2 = f".pt 模型結果：{severity_labels[class_idx_pt]} (等級 {class_idx_pt})\n信心度：{conf_pt:.4f}\n"
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