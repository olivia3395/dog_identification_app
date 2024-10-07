from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model input
    img_array = preprocess_input(img_array)  # Use the preprocess_input function for VGG16
    return img_array

def predict_and_visualize(model, image_paths):
    plt.figure(figsize=(10, 10))
    
    for i, img_path in enumerate(image_paths):
        # Preprocess image
        img_array = preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Load image for visualization
        img = Image.open(img_path)

        # Display the image and prediction
        plt.subplot(2, 2, i+1)  # Create a 2x2 grid of subplots
        plt.imshow(img)
        plt.title(f"Predicted Class: {predicted_class}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()