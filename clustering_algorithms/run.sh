
python3 correlation_algorithms/whole_correlation/correlation_mip.py ../inputs/real_weights/pytorch_weights_36x36_1.pt > correlation_algorithms/whole_correlation/runs/fc1_8000.txt

python3 correlation_algorithms/block_correlation/block_correlation_mip.py ../inputs/real_weights/fc1_81x81.pt > correlation_algorithms/block_correlation/runs/fc1_8000.txt

python3 manhattan_algorithms/manhattan_L1_distance_mip.py ../inputs/real_weights/fc1_81x81.pt > manhattan_algorithms/runs/fc1_8000.txt

