import os
import torch

# declare which gpu device to use
cuda_device = '0'

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.FloatTensor(256,1024,block_mem).to(torch.device(f"cuda:{cuda_device}"))
    del x

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
occumpy_mem('0')
occumpy_mem('1')