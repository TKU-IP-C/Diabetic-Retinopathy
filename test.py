import torch
print(torch.cuda.is_available())  # 應該要返回 True
print(torch.cuda.get_device_name(0))  # 會顯示你的顯卡型號