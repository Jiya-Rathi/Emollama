import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU:", torch.cuda.get_device_name(0))
    a = torch.rand(10000, 10000).cuda()
    b = torch.mm(a, a)
    print("GPU matrix multiplication successful.")
else:
    print("CUDA not available.")
