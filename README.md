# Ganspix2pix

This project implements a Generative Adversarial Network (GAN) based on the Pix2Pix architecture for coloring grayscale images. The Pix2Pix framework is a conditional GAN designed for image-to-image translation tasks, making it highly suitable for applications such as colorization, image inpainting, and style transfer.

**Overview**

The goal of this project is to use the Pix2Pix GAN to generate realistic colored versions of grayscale images. By training the model on paired datasets (grayscale and corresponding colored images), the generator learns to add plausible and visually appealing color to grayscale inputs.

**Features**

Implementation of Pix2Pix GAN architecture.

Training pipeline for grayscale-to-color image translation.

Visualization of training progress and generated results.

Support for custom datasets.

Preprocessing pipeline for preparing grayscale and colored image pairs.

**Project Structure**

**project-folder/**
|— pix2pix-gan-coloring-images.ipynb  # Jupyter Notebook with the full implementation
|— README.md                           # Project documentation

**Dependencies**

This project requires the following Python libraries:

Python 3.8+

TensorFlow/Keras

NumPy

Matplotlib

PIL (Pillow)

**You can install these dependencies using the following command:**

pip install tensorflow numpy matplotlib pillow

**Dataset**

The model requires paired grayscale and color images for training. You can use publicly available datasets or prepare your own. Make sure the images are preprocessed to have the same dimensions and are normalized appropriately.

**Example datasets:**

COCO-Stuff Dataset

Custom datasets with grayscale and color pairs

Place your dataset in the data/ directory, organized into train/ and test/ subfolders.

Usage

**Clone the repository:**

git clone https://github.com/your-username/pix2pix-gan-coloring-images.git
cd pix2pix-gan-coloring-images

Prepare your dataset and place it in the data/ directory.

Open the Jupyter Notebook:

jupyter notebook pix2pix-gan-coloring-images.ipynb

Follow the instructions in the notebook to train the model and visualize results.

Trained models will be saved in the models/ directory, and generated images will be saved in the results/ directory.

Results

The trained Pix2Pix GAN produces realistic colored images from grayscale inputs. Below is an example of the model's output:

Grayscale Input

![image](https://github.com/user-attachments/assets/365bbfaa-ee98-46b3-9cd9-226d38a6da33)

Generated Color

![image](https://github.com/user-attachments/assets/b31e02bc-36ea-484f-9799-1b9e74ebf37f)

Ground Truth

![image](https://github.com/user-attachments/assets/3bf7af6b-7e49-41e6-9077-be85eae1c19a)

**Customization**

Modify the hyperparameters such as learning rate, batch size, and number of epochs in the notebook to improve performance.

Use your own datasets by preparing paired grayscale and color images.

Experiment with different image sizes to find the optimal resolution for your application.

References

Pix2Pix Paper: "Image-to-Image Translation with Conditional Adversarial Networks"

TensorFlow Pix2Pix Tutorial
