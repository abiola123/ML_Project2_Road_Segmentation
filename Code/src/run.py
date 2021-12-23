from transfer_learning_unet import create_fit_pretrained_model 
from network_accuracy_testing import networkAccuracyTesting

def run():

    PATH_TO_YOUR_TRAINING_SET= "put path here"
    PATH_TO_YOUR_TRAINING_LABELS = "put path here"
    PATH_WHERE_MODEL_SHOULD_BE_SAVED = "put model path here"
    
    #create and train model
    create_fit_pretrained_model(PATH_TO_YOUR_TRAINING_SET,PATH_TO_YOUR_TRAINING_LABELS,Encoder="resnet18")

    #evaluate model and generate csv

    TEST_IMAGES_PATH = "put path to test images here"
    SUBMISSION_PATH = "put path where csv file should endhere"
    SUBMISSION_NAME = "put submission name"
    PREDICTION_PATH = "put path where predictions (masks) should end up"
    networkAccuracyTesting(PATH_WHERE_MODEL_SHOULD_BE_SAVED + "resnet18" + ".h5",TEST_IMAGES_PATH,SUBMISSION_PATH,SUBMISSION_NAME,PREDICTION_PATH)

    #search for the csv at specified path => done



run()