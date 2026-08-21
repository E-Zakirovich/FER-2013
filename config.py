# dataset paths
dataset_path = "./data"

# data pipeline settings
image_size = 48
amount_of_flip = 0.5
angle_range = 15
mean = 0.5
std = 0.5
seed = 42
batch_size = 16
num_workers = 2
train_split = 0.8
validation_split = 0.1
test_split = 0.1
total_channels = 1

# convolutional neural networks parameters
in_out_channels = [1, 32, 64, 128]
kernel_size = 3
stride = 1
padding = 1
kernel_and_stride_size_for_pooling = 2
input_layer = 6 * 6 * 128
hidden_layer = 512
output_layer = 7
dropout = 0.3

# train parameters
learning_rate = 0.001