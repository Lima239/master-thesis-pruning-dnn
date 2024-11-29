import torch

def generate_and_save_tensor(file_path):
    data = torch.load("../../substitution_into_model/fc1.pt")
    #matrix = data["decoder.layers.1.self_attn.v_proj.weight"]

    matrix = data[:81, :81]
    torch.save(matrix, file_path)
    print(matrix.shape)
    print(f'Tensor saved to {file_path}')

if __name__ == "__main__":
    output_file = 'fc1_81x81.pt'

    generate_and_save_tensor(output_file)