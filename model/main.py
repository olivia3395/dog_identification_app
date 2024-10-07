from data_loader import download_and_extract_datasets, prepare_data_generators
from model import create_vgg16_model, load_trained_model
from train import train_model
from predict import predict_and_visualize
import os

# Step 1: Download and Extract Dataset
if not os.path.exists('dogImages/dogImages'):
    download_and_extract_datasets()

# Step 2: Prepare Data Generators
train_generator, valid_generator, test_generator = prepare_data_generators()

# Step 3: Train VGG16 Model
vgg16_model = create_vgg16_model(train_generator)
train_model(vgg16_model, train_generator, valid_generator)

# Step 4: Load the trained model and run predictions
model = load_trained_model()
test_image_paths = ['./test_data/dog1.jpg', './test_data/dog2.jpg', './test_data/dog3.jpg']
predict_and_visualize(model, test_image_paths)
