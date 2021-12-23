from helpers import *
import cv2
import tensorflow as tf
import os
import numpy as np

def networkAccuracyTesting(model_path,test_images_path,submission_path,submission_filename,prediction_path):

  #Load saved model from h5 file
  print("Loading model...")
  model = tf.keras.models.load_model(model_path,compile = False)

  print("Model loaded")
  print(model.summary())
  #Load test images
  test_image_dir = test_images_path
  files = os.listdir(test_images_path)

  nb_test_imgs = len(files)
  
  #The test images are loaded from 1 to 50
  test_imgs = [load_image(test_image_dir + "test_" + str(i+1) + ".png") for i in range(nb_test_imgs)]
  test_imgs = np.asarray(test_imgs)

  X_test = test_imgs

  #Sanity check
  print("X_test shape: ",X_test.shape)


  #Generate masks for all test images and show them next to the images to see how well our model works
  print("Generating masks for each test image....")
  fig = plt.figure(figsize=(250, 250))
  for i in range(len(X_test)):
    elem = X_test[i]
    elem = cv2.resize(elem, (400,400), interpolation = 0)
    elem = elem.reshape((1,400,400,3))

    pred = model.predict(elem.reshape((1,400,400,3)), verbose = 2)

    pred = ((pred > 0.25).astype(np.uint8)).reshape((400,400))

    pred = cv2.resize(pred, (608,608), interpolation = 0)
    
    plt.imsave(os.path.join(prediction_path + "test_" + str(i+1) + ".png"),pred,cmap='Greys_r')


  print("Generating CSV for submission...")
  #Generate Submission File From Masks
  submission_filename = submission_filename
  image_filenames = []
  for i in range(1, 51):
    image_filename = prediction_path + "test_" + str(i) + '.png'
    image_filenames.append(image_filename)
  masks_to_submission(submission_filename, *image_filenames)

  print("Submission file saved at "+ submission_path)

networkAccuracyTesting("/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/models/road_semgentation_unet12tn.h5","/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/test_set_images/","/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/submissions/","sub1.csv","/Users/abiola/Documents/EPFL/ML_Project2_Road_Segmentation/Ressources/predictions/")