import torch

def generate_and_save_tensor(file_path):
    data = torch.load("/pytorch_model.bin")
    matrix = data["decoder.layers.1.self_attn.v_proj.weight"]

    torch.save(matrix, file_path)
    print(f'Tensor saved to {file_path}')

if __name__ == "__main__":
    output_file = 'pytorch_weights.pt'

    generate_and_save_tensor(output_file)