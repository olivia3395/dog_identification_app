# Dog Identification App

This project is a Dog Breed Identification application built with TensorFlow/Keras. The model leverages a pre-trained VGG16 model for feature extraction, followed by a custom classifier to predict the breed of dogs from images.

## Project File

The project is contained in a single Jupyter notebook:

- **DogIdentificationApp.ipynb**: Contains the full implementation of the app, including the model loading, image preprocessing, prediction, and visualization of results.

## Dataset

The app uses a pre-trained VGG16 model and relies on images for testing. You can add your dog images into a folder named `test_data` and use this application to predict their breeds.

## Prerequisites

Ensure you have the following packages installed:

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `Pillow`

Install the required packages using pip:

```bash
pip install tensorflow keras numpy matplotlib Pillow
```

## Running the Application

1. Clone this repository to your local machine.
2. Download and place your test dog images.
3. Open `DogIdentificationApp.ipynb` in Jupyter Notebook.
4. Run the notebook to make predictions on your images.

## Usage

Upload dog images to the `test_data` folder. The notebook will process the images, extract features using VGG16, and make predictions using a pre-trained classifier. The results will be visualized in a grid, displaying the image along with the predicted breed.

## License

This project is licensed under the MIT License.
