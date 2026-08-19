# dataset paths
dataset_path = "./data"

# augmentation settings
image_size = 128
amount_of_flip = 0.5
angle_range = 15
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
seed = 42
batch_size = 16
num_workers = 2
train_split = 0.8
validation_split = 0.1
test_split = 0.1