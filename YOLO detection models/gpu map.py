import torch

print(torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
device = "cuda:0"
new = torch.cuda.get_device_name(device=device)
print("device being used: ", new)