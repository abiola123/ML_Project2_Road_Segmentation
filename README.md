# Project : Road Segmentation

## Introduction

This repository shows and documents our attempt at solving the EPFL Road Segmentation Challenge on AICrowd: https://www.aicrowd.com/challenges/epfl-ml-road-segmentation

This project has been developed in the context of the Machine Learning course CS-433 at EPFL.

The aim of this project is to train a classifier to differentiate roads from background noise on satellite images.

Further information is available in our report.

## Project structure

The project is structured as follows:

```
├── src # Source code directory
| ├── run.py # File that runs the whole training pipeline and generates the submission file for the competition
│ ├── custom_unet.py # Shell for creating and tuning our U-Net architecture
│ ├── data_expansion.py # Generate new data
│ ├── helper.py # General functions
│ ├── network_accuracy_testing.py # Method that generates submissions
│ ├── transfer_learning_unet.py # Shell for creating and tuning Pretrained models
├── KerasUNet.ipynb # Used for experimental purposes (hyperparameter optimization and visualization of the training process)
```

## Requirements

The code has been trained and tested using:

- TensorFlow
- Keras
- Numpy
- Matplotlib

## Instructions

```
git clone <repo_url> // clone the repo
cd src
python run.py (To be used for course evaluation)
```

## Training Hardware

The training and tests that we did were done using Google Colab on the Pro plan. This gave us access to the following resources:

- RAM: 20 GB
- CPU: 2 vCPU
- GPU: Nvidia Tesla P100

### Authors

- Abiola Adeye
- Abdeslam Guessous
- Aya Rahmoun
