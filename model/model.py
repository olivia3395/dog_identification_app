from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def create_vgg16_model(train_generator):
    # Load the pre-trained VGG16 model, excluding the fully connected layers (top)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Build the model on top of the base
    vgg16_model = Sequential()
    vgg16_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
    vgg16_model.add(Dense(train_generator.num_classes, activation='softmax'))

    return vgg16_model

def load_trained_model():
    return load_model("./VGG16_model.keras")