import os
import zipfile
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def download_and_extract_datasets():
    url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip'
    dataset_path = 'dogImages.zip'

    # Download dataset
    if not os.path.exists(dataset_path):
        response = requests.get(url)
        with open(dataset_path, 'wb') as f:
            f.write(response.content)

    # Unzip the dataset
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall('dogImages')

def prepare_data_generators():
    # Create image data generators for train, validation, and test datasets
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        'dogImages/dogImages/train',
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
    )

    valid_generator = datagen.flow_from_directory(
        'dogImages/dogImages/valid',
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        'dogImages/dogImages/test',
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
    )

    return train_generator, valid_generator, test_generator