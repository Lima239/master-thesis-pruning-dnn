import torch

def generate_and_save_tensor(size, file_path):
    row = torch.arange(1, size + 1, dtype=torch.float32)
    matrix = row.repeat(size, 1)

    torch.save(matrix, file_path)
    print(f'Tensor of size {size} saved to {file_path}')

if __name__ == "__main__":
    matrix_size = 9
    output_file = '9x9.pt'

    generate_and_save_tensor(matrix_size, output_file)
