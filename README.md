# Project : Road Segmentation

## Introduction

Road segmentation, which is the process of identifying roads on satellite imagery, is part of Image Segmentation. 
This project has been developed in the context of the Machine Learning course CS-433 at EPFL. The aim of this project is to
train a classifier to differentiate roads from background on satellite images.

## Project structure

The project is structured as follows:

```
.
├── Course_Example # Code provided by the course      	
├── src # Source code directory
|   ├── run.py # Files that generate our best submissions 
│   ├── custom_unet.py # Shell for creating and tuning our U-Net architecture
│   ├── data_expansion.py # Generate new data
│   ├── helper.py # General functions 
│   ├── network_accuracy_testing.py # Method that generates submissions
│   ├── transfer_learning_unet.py # Shell for creating and tuning Pretrained models
├── KerasUNet.ipynb # Used for experimental purposes (hyperparameter optimization and visualization of the training process)
```

## Requirements

The code has been trained using:
- TensorFlow 
- Keras
- Numpy
- Matplotlib

### Authors
- Abiola Adeye
- Abdeslam Guessous
- Aya Rahmoun
