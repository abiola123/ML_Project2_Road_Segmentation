from helpers import * 
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import scipy.ndimage 
import pandas
import tensorflow as tf
import cv2
from PIL import Image
import time
import h5py


############################## FLIP PIPELINE ########################################
def flip_images_lr(imgs,gt_imgs):
    n = len(imgs)
    imgs_flipped = [np.fliplr(imgs[i]) for i in range(n)]
    gt_imgs_flipped = [np.fliplr(gt_imgs[i]) for i in range(n)]
    return imgs_flipped, gt_imgs_flipped

def flip_images_ud(imgs,gt_imgs):
    n = len(imgs)
    imgs_flipped = [np.flipud(imgs[i]) for i in range(n)]
    gt_imgs_flipped = [np.flipud(gt_imgs[i]) for i in range(n)]
    return imgs_flipped, gt_imgs_flipped



def flip_pipeline(imgs,gt_imgs):
    imgs_flipped_lfr, gt_imgs_flipped_lfr = flip_images_lr(imgs,gt_imgs)
    imgs_flipped_ud, gt_imgs_flipped_ud = flip_images_ud(imgs,gt_imgs)
   
    return imgs_flipped_lfr + imgs_flipped_ud,gt_imgs_flipped_lfr+gt_imgs_flipped_ud

########################################################################################


############################## ROTATION PIPELINE ########################################
def rotate(imgs,gt_imgs,angle):
    n = len(imgs)
    imgs_rot = [scipy.ndimage.interpolation.rotate(imgs[i], angle, axes=(1, 0), reshape=False, output=None, order=3, mode='reflect', cval=0.0, prefilter=True) for i in range(n)]
    #imgs_rot_resized = [cv2.resize(imgs_rot[i], (400,400), interpolation = 0) for i in range(n)]
    gt_imgs_rot = [scipy.ndimage.interpolation.rotate(gt_imgs[i], angle, axes=(1, 0), reshape=False, output=None, order=3, mode='reflect', cval=0.0, prefilter=True) for i in range(n)]
    #gt_imgs_rot_resized = [cv2.resize(gt_imgs_rot[i], (400,400), interpolation = 0) for i in range(n)]
    return imgs_rot,gt_imgs_rot
    

    
def rotatation_pipeline(imgs,gt_imgs,nb_rotations):
    
    imgs_output = []
    gt_imgs_output = []
    stepSize = 360/nb_rotations
    angle = stepSize
    for i in range(nb_rotations):
        print(i, "out of ", nb_rotations)
        imgs_rot_tmp, gt_imgs_rot_tmp = rotate(imgs,gt_imgs,angle)
        imgs_output = imgs_output + imgs_rot_tmp
        gt_imgs_output = gt_imgs_output + gt_imgs_rot_tmp
        angle += stepSize
   
    return imgs_output,gt_imgs_output
###############################################################################################


def data_expansion(root_ressources_dir,h5_file_path = None):
    image_dir = root_ressources_dir + "images/"
    files = os.listdir(image_dir)
    n = len(files)
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    gt_dir = root_ressources_dir + "groundtruth/"
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    imgs_flipped, gt_imgs_flipped = flip_pipeline(imgs,gt_imgs)

    imgs_out = imgs + imgs_flipped
    gt_imgs_out = gt_imgs + gt_imgs_flipped

    print(np.array(imgs_out).shape)
    print(np.array(gt_imgs_out).shape)
    print("\n\n********************* flip step done *************************")

    #rotate by steps of 45 degress 
  
    imgs_rotated, gt_imgs_rotated = rotatation_pipeline(imgs_out,gt_imgs_out,8)

    imgs_out = imgs_rotated
    gt_imgs_out = gt_imgs_rotated

    print(np.array(imgs_out).shape)
    print(np.array(gt_imgs_out).shape)
    print("\n\n********************* rotation step done *************************")


    if h5_file_path != None:
        print("\n\nWriting IMAGES and LABELS to given path......")
        
        with h5py.File(h5_file_path + 'IMGS.h5', 'w') as hf:
            hf.create_dataset("IMGS",  data=imgs_out)
        print("WRITING IMGS TO H5 DONE")

        with h5py.File(h5_file_path + 'LABELS.h5', 'w') as hf:
            hf.create_dataset("LABELS",  data=gt_imgs_out)
        print("WRITING LABELS TO H5 DONE")

    return imgs_out,gt_imgs_out




data_expansion("/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/training/", h5_file_path= "/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/")
