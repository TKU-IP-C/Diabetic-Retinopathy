import torch

ckpt = torch.load("yolo26n.pt", map_location="cpu")

print(ckpt.keys())

print("Ultralytics version:", ckpt.get("version"))
print("Train args:", ckpt.get("train_args"))
print("YAML:", ckpt.get("yaml"))
