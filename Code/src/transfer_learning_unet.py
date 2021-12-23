from helpers import *
import tensorflow as tf
import os
import sys
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import h5py
import time
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Conv2D

learning_rate = 0.0001


#* The loss function that we are going to use is dice_loss/focal_loss because ....



def create_fit_pretrained_model(X_path,Y_path, Encoder,  image_dimensions = (400,400),activation = "sigmoid",optimizer = tf.keras.optimizers.Adam(learning_rate),EPOCHS = 100,VALIDATION_SPLIT = 0.3,BATCH_SIZE = 32,model_save_location = "./"):
    '''CREATE AND TRAIN A UNET USING PRETRAINED WEIGHTS FROM VARIOUS WELL KNOWN IMAGE SEGMENTATION ARCHITECTURES: https://github.com/qubvel/segmentation_models to see all the available architectures'''

    #X and Y are given as h5 files. Allows to load images faster
    startTime = time.time()
    with h5py.File(X_path, 'r') as hf:
        imgs = hf['IMGS'][:]

    print("--- Time to load images: %s seconds ---" % (time.time() - startTime))


    startTime = time.time()
    with h5py.File(Y_path, 'r') as hf:
        gt_imgs = hf['LABELS'][:]

    print("--- Time to load labels: %s seconds ---" % (time.time() - startTime))

    X_train = np.asarray(imgs)
    Y_train = np.asarray(gt_imgs).reshape((len(gt_imgs),image_dimensions[0],image_dimensions[1],1))

    X_train_nxt= []
    for i in range(len(X_train)):
        elem = cv2.resize(X_train[i], (416,416), interpolation = 0)
        X_train_nxt.append(elem)

    Y_train_nxt= []
    for i in range(len(Y_train)):
        elem = cv2.resize(Y_train[i], (416,416), interpolation = 0)
        Y_train_nxt.append(elem)

    X_train = np.asarray(X_train_nxt)
    Y_train = np.asarray(Y_train_nxt).reshape((len(Y_train),416,416,1))

    #Shuffle our training set and the labels
    idx = np.random.RandomState().permutation(len(X_train))
    X_train,Y_train = X_train[idx], Y_train[idx]

    #Instantiate loss function and metrics for training our model 
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25,0.75]))
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = [sm.metrics.IOUScore(threshold=0.5),sm.metrics.FScore(threshold=0.5)]


    ######################## Callbacks for fit method ###################################

    #1. checkpoints to save timestamps of model during training
    checkpoints = tf.keras.callbacks.ModelCheckpoint(model_save_location + Encoder + ".h5", verbose=1, save_best_only = True)

    #2. eraly stopping callback to avoid useless epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, monitor="val_f1-score",restore_best_weights = True,verbose=1)
    #val_loss

    #3. tensorboard to visulaize various statistics during training
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

    #4. dynamically adjust learning rate when training is not progressing well

    dynamic_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, verbose=1, epsilon=1e-4, min_lr=0.000001)

    callbacks = [checkpoints,dynamic_lr,tensorboard,early_stopping]

    #######################################################################################


    print("Creating model .....")
    resnet_model = sm.Unet(Encoder, encoder_weights="imagenet", activation=activation)

    resnet_model.compile(optimizer, total_loss, metrics = metrics)

    print(resnet_model.summary())


    print("Starting model training, model is going to be saved at ", model_save_location+ Encoder+".h5")

    resnet_results = resnet_model.fit(X_train,Y_train,batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split =VALIDATION_SPLIT, verbose = 1,callbacks=callbacks)

create_fit_pretrained_model("/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/training/IMGS_100.h5","/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/training/LABELS_100.h5", "resnext50", model_save_location="/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/models/")