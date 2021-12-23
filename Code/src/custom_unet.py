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


# MODEL_CONSTANTS

LEARNING_RATE = 1e-3


############################# HELPERS ###########################################################
#Source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#Source: https://towardsdatascience.com

from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  
  return iou


def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)  

##############################################################################################   


def create_fit_custom_unet(X_path,Y_path, image_dimensions = (400,400),DROPOUT_PERCENTAGE = 0.2,CONVOLUTION_WINDOW = (4,4),POOLING_WINDOW = (2,2),DECONVOLUTION_WINDOW = (2,2),optimizer = tf.keras.optimizers.Adam(LEARNING_RATE),EPOCHS = 100,VALIDATION_SPLIT = 0.3,BATCH_SIZE = 32,loss=soft_dice_loss,model_save_location = "./"):
    
    #X and Y are given as h5 files. Allows to load images faster
    startTime = time.time()
    with h5py.File(X_path, 'r') as hf:
        imgs = hf['IMGS'][:]

    print("--- Time to load images: %s seconds ---" % (time.time() - startTime))

    print(image_dimensions[0],image_dimensions[1])


    startTime = time.time()
    with h5py.File(Y_path, 'r') as hf:
        gt_imgs = hf['LABELS'][:]

    print("--- Time to load labels: %s seconds ---" % (time.time() - startTime))

    X_train = np.asarray(imgs)
    Y_train = np.asarray(gt_imgs).reshape((len(gt_imgs),400,400,1))
    
    #Shuffle our training set and the labels
    idx = np.random.RandomState().permutation(len(X_train))
    X_train,Y_train = X_train[idx], Y_train[idx]
    
    
    # Model
    #dropouts were added to avoid overfitting

    #Encoder Path
    inputs = tf.keras.layers.Input((400,400,3))

    #transform input which can be integer value into floating point values
    #transform = tf.keras.layers.Lambda(lambda x : x / 1.0)(inputs)
    transform = tf.keras.layers.Normalization(axis=-1)(inputs)
    #first convolution
    c1 = tf.keras.layers.Conv2D(32, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(transform)
    c1 = tf.keras.layers.BatchNormalization() (c1)
    c1 = tf.keras.layers.LeakyReLU(alpha=0.1)(c1)
    c1 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(c1)
    c1 = tf.keras.layers.Conv2D(32, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(c1)
    c1 = tf.keras.layers.BatchNormalization() (c1)
    c1 = tf.keras.layers.LeakyReLU(alpha=0.1)(c1)

    #first pooling
    p1 = tf.keras.layers.MaxPooling2D(POOLING_WINDOW)(c1)

    #second convolution
    c2 = tf.keras.layers.Conv2D(64,CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(p1)
    c2 = tf.keras.layers.BatchNormalization() (c2)
    c2 = tf.keras.layers.LeakyReLU(alpha=0.1)(c2)
    c2 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(c2)
    c2 = tf.keras.layers.Conv2D(64, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(c2)
    c2 = tf.keras.layers.BatchNormalization() (c2)
    c2 = tf.keras.layers.LeakyReLU(alpha=0.1)(c2)

    #second pooling
    p2 = tf.keras.layers.MaxPooling2D(POOLING_WINDOW)(c2)

    #third convolution
    c3 = tf.keras.layers.Conv2D(128, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(p2)
    c3 = tf.keras.layers.BatchNormalization() (c3)
    c3 = tf.keras.layers.LeakyReLU(alpha=0.1)(c3)
    c3 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(c3)
    c3 = tf.keras.layers.Conv2D(128, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(c3)
    c3 = tf.keras.layers.BatchNormalization() (c3)
    c3 = tf.keras.layers.LeakyReLU(alpha=0.1)(c3)

    #third pooling
    p3 = tf.keras.layers.MaxPooling2D(POOLING_WINDOW)(c3)

    #fourth convolution
    c4 = tf.keras.layers.Conv2D(256, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(p3)
    c4 = tf.keras.layers.BatchNormalization() (c4)
    c4 = tf.keras.layers.LeakyReLU(alpha=0.1)(c4)
    c4 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(c4)
    c4 = tf.keras.layers.Conv2D(256, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(c4)
    c4 = tf.keras.layers.BatchNormalization() (c4)
    c4 = tf.keras.layers.LeakyReLU(alpha=0.1)(c4)

    #fourth pooling
    p4 = tf.keras.layers.MaxPooling2D(POOLING_WINDOW)(c4)

    #fifth convolution
    c5 = tf.keras.layers.Conv2D(512, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(p4)
    c5 = tf.keras.layers.BatchNormalization() (c5)
    c5 = tf.keras.layers.LeakyReLU(alpha=0.1)(c5)
    c5 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(c5)
    c5 = tf.keras.layers.Conv2D(512, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(c5)
    c5 = tf.keras.layers.BatchNormalization() (c5)
    c5 = tf.keras.layers.LeakyReLU(alpha=0.1)(c5)


    #Decoder Path
    #at each step we concatenate the results from the encoder path with the results from the deocder path => improves image segmentation


    #first upsampling
    u1 = tf.keras.layers.Conv2DTranspose(256, DECONVOLUTION_WINDOW, strides = (2,2), padding="same")(c5)
    u1 = tf.keras.layers.concatenate([u1, c4])
    u1 = tf.keras.layers.Conv2D(256, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u1)
    u1 = tf.keras.layers.BatchNormalization() (u1)
    u1 = tf.keras.layers.LeakyReLU(alpha=0.1)(u1)
    u1 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(u1)
    u1 = tf.keras.layers.Conv2D(256, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u1)
    u1 = tf.keras.layers.BatchNormalization() (u1)
    u1 = tf.keras.layers.LeakyReLU(alpha=0.1)(u1)

    #second upsampling
    u2 = tf.keras.layers.Conv2DTranspose(128, DECONVOLUTION_WINDOW, strides = (2,2), padding="same")(u1)
    u2 = tf.keras.layers.concatenate([u2, c3])
    u2 = tf.keras.layers.Conv2D(128, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u2)
    u2 = tf.keras.layers.BatchNormalization() (u2)
    u2 = tf.keras.layers.LeakyReLU(alpha=0.1)(u2)
    u2 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(u2)
    u2 = tf.keras.layers.Conv2D(128, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u2)
    u2 = tf.keras.layers.BatchNormalization() (u2)
    u2 = tf.keras.layers.LeakyReLU(alpha=0.1)(u2)

    #third upsampling
    u3 = tf.keras.layers.Conv2DTranspose(64, DECONVOLUTION_WINDOW, strides = (2,2), padding="same")(u2)
    u3 = tf.keras.layers.concatenate([u3, c2])
    u3 = tf.keras.layers.Conv2D(64, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u3)
    u3 = tf.keras.layers.BatchNormalization() (u3)
    u3 = tf.keras.layers.LeakyReLU(alpha=0.1)(u3)
    u3 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(u3)
    u3 = tf.keras.layers.Conv2D(64, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u3)
    u3 = tf.keras.layers.BatchNormalization() (u3)
    u3 = tf.keras.layers.LeakyReLU(alpha=0.1)(u3)

    #fourth upsampling
    u4 = tf.keras.layers.Conv2DTranspose(32, DECONVOLUTION_WINDOW, strides = (2,2), padding="same")(u3)
    u4 = tf.keras.layers.concatenate([u4, c1])
    u4 = tf.keras.layers.Conv2D(32, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u4)
    u4 = tf.keras.layers.BatchNormalization() (u4)
    u4 = tf.keras.layers.LeakyReLU(alpha=0.1)(u4)
    u4 = tf.keras.layers.Dropout(DROPOUT_PERCENTAGE)(u4)
    u4 = tf.keras.layers.Conv2D(32, CONVOLUTION_WINDOW,  kernel_initializer="he_normal",padding="same")(u4)
    u4 = tf.keras.layers.BatchNormalization() (u4)
    u4 = tf.keras.layers.LeakyReLU(alpha=0.1)(u4)



    out = tf.keras.layers.Conv2D(1, (1,1), activation="sigmoid")(u4)


    model = tf.keras.Model(inputs = [inputs], outputs=[out])


    model.compile(optimizer=optimizer,loss=loss,metrics=[f1_m])
    print(model.summary())


    ######################## Callbacks for fit method ###################################

    #1. checkpoints to save timestamps of model during training
    checkpoints = tf.keras.callbacks.ModelCheckpoint(model_save_location + "custom_unet" + ".h5", verbose=1, save_best_only = True)

    #2. eraly stopping callback to avoid useless epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, monitor="val_f1-score",restore_best_weights = True,verbose=1)
    #val_loss

    #3. tensorboard to visulaize various statistics during training
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

    #4. dynamically adjust learning rate when training is not progressing well

    dynamic_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, verbose=1, epsilon=1e-4, min_lr=0.000001)

    callbacks = [checkpoints,dynamic_lr,tensorboard,early_stopping]

    #######################################################################################

    results = model.fit(X_train,Y_train, validation_split = 0.2,batch_size =5, epochs = EPOCHS, callbacks = callbacks)




create_fit_custom_unet("/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/training/IMGS_100.h5","/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/training/LABELS_100.h5", "resnext50", model_save_location="/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/models/")