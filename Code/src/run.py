from transfer_learning_unet import create_fit_pretrained_model 
from network_accuracy_testing import networkAccuracyTesting

def run():

    PATH_TO_YOUR_TRAINING_SET= "put path here"
    PATH_TO_YOUR_TRAINING_LABELS = "put path here"
    PATH_WHERE_MODEL_SHOULD_BE_SAVED = "put model path here"
    
    #create and train model
    create_fit_pretrained_model(PATH_TO_YOUR_TRAINING_SET,PATH_TO_YOUR_TRAINING_LABELS,Encoder="resnet18")

    #evaluate model and generate csv
    networkAccuracyTesting(PATH_WHERE_MODEL_SHOULD_BE_SAVED + "resnet18" + ".h5",)



run()