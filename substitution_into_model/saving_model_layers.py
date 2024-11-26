import torch
import torch.nn as nn
import timm

model_name = "test_vit3.r160_in1k"
model = timm.create_model(model_name, pretrained=True)

select_all_linear = lambda n, n2, m: type(m) == nn.Linear and "head" not in n and "head" not in n2
select_only_attn_proj = lambda n, n2, m: type(m) == nn.Linear and "head" not in n and "head" not in n2 and "proj" in n2

layer_matrices = {}

def capture_matrice(module,name):
    matrix = module.weight.data
    save_path = f"{name}.pt"
    torch.save(matrix, save_path)
    layer_matrices[name] = matrix.cpu()

for n, m in model.named_modules():
    for n2, m2 in m.named_modules():
        if "." not in n2 and select_all_linear(n, n2, m2) and len(n2) > 0:
            #capture_matrice(m2, n2)
            print(n2)
            print(m2.weight.data.shape)

