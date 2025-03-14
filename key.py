import torch

model_path = "model_zoo/team34_Unet.pth"
checkpoint = torch.load(model_path)
print(checkpoint.keys())