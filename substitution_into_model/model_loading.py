import os

# swapped numbering as in nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from functools import partial

IMAGENET_PATH = "/data/imagenet/imagenet/"

device = torch.device("cuda")
amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)
model_name = "test_vit3.r160_in1k"
model = timm.create_model(model_name, pretrained=True)
model.cuda()

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input, target in tqdm(val_loader):
            with amp_autocast():
                output = model(input)
                correct += (target == output.argmax(dim=1)).sum().item()
                total += target.numel()
    print("val acc", correct / total)


val_dataset = timm.data.create_dataset(
    name="imagenet",
    split="validation",
    root=IMAGENET_PATH
)

val_loader = timm.data.create_loader(
    val_dataset,
    input_size=data_config['input_size'],
    batch_size=128,
    use_prefetcher=True,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=8,
    crop_pct=data_config["crop_pct"],
    crop_mode=data_config['crop_mode'],
    crop_border_pixels=False,
    pin_memory=True,
    device=device,
)

for bx, by in val_loader:
    print(bx.shape, by.shape)
    print(by)
    break

train_dataset = timm.data.create_dataset(
    name="imagenet",
    split="train",
    root=IMAGENET_PATH
)

train_loader = timm.data.create_loader(
    train_dataset,
    input_size=data_config['input_size'],
    batch_size=128,
    use_prefetcher=True,
    interpolation="random",
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=12,
    crop_pct=data_config["crop_pct"],
    crop_mode=data_config['crop_mode'],
    crop_border_pixels=False,
    pin_memory=True,
    device=device,
    is_training=True,
)

def update_cov(m, i, o):
    x = i[0].detach().flatten(0, -2).float()
    with torch.autocast(device_type="cuda", enabled=False):
        m.XX.data += x.square().sum(dim=0)

# n - name of the module in model
# n2 - string name of the layer
# m - instance of the layer
select_all_linear = lambda n, n2, m: type(m) == nn.Linear and "head" not in n and "head" not in n2
select_only_attn_proj = lambda n, n2, m: type(m) == nn.Linear and "head" not in n and "head" not in n2 and "proj" in n2


