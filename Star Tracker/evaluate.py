#Load model
from keras.models import load_model
model = load_model('C:/PythonPrograms/GitClones/StarTracker_Thesis/Star Tracker/Trained_Mini_StarTracker.h5')

#Predict function
import os
import cv2
import numpy as np
from keras.preprocessing import image

def predict_class(path):
    predict_results = []
    for filename in os.listdir(path):
        test_image = image.load_img(
            os.path.join(path,filename),
            target_size=(64,64)
            )
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        result = cnn.predict(test_image)
        predict_results.append(result)
    return predict_results

