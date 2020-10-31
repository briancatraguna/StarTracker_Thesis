#Load model
from keras.models import load_model
model = load_model('C:/PythonPrograms/GitClones/StarTracker_Thesis/Star Tracker/Trained_Mini_StarTracker.h5')

#Predict function
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

def predict_class(class_no):
    directory = 'C:/PythonPrograms/GitClones/StarTracker_Thesis/Star Tracker/dataset/test/'
    class_no = str(class_no)+'/'
    path = os.path.join(directory,class_no)
    predict_results = []
    for filename in os.listdir(path):
        test_image = image.load_img(
            os.path.join(path,filename),
            target_size=(64,64)
            )
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        result = model.predict(test_image)
        result = result.reshape(8)
        predict_results.append(result)
    return predict_results

def get_accuracy(predict_results,class_no):
    right = 0
    wrong = 0
    for result in predict_results:
        if result[class_no] == 1:
            right+=1
        else:
            wrong+=1
    accuracy = right/(right+wrong)
    return accuracy