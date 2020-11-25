#Load model
from keras.models import load_model
model = load_model('C:/PythonPrograms/Deep Learning Models/net_model.h5')

#Predict function
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import time
from keras import backend as K

def predict_class(class_no):
    directory = 'dataset_for_net_algo/test/'
    class_no = str(class_no)+'/'
    path = os.path.join(directory,class_no)
    predict_results = []
    time_total = []
    for filename in os.listdir(path):
        start_time = time.time()
        test_image = image.load_img(
            os.path.join(path,filename),
            target_size=(64,64)
            )
        test_image = image.img_to_array(test_image)
        test_image = test_image/test_image.max()
        test_image = np.expand_dims(test_image,axis=0)
        result = model.predict(test_image)
        result = result.reshape(8)
        final_time = time.time()
        delta_t = final_time - start_time
        time_total.append(delta_t)
        predict_results.append(result)
    return predict_results,time_total

def get_accuracy(predict_results,class_no):
    right = 0
    wrong = 0
    not_fully_confident = 0
    for result in predict_results:
        if result[class_no] == 1:
            right+=1
        else:
            wrong+=1
        for neuron in result:
            if neuron == 1:
                continue
            elif neuron != 0:
                not_fully_confident+=1
                break
    accuracy = right/(right+wrong)
    return accuracy,not_fully_confident

results = []
accuracy_list = []
time_ave = []
for i in range(8):
    result,time_perclass = predict_class(i)
    results.append(result)
    time_ave.append(sum(time_perclass)/3600)
    accuracy,confidence_number = get_accuracy(results[i],i)
    accuracy_list.append(accuracy)
    print("The accuracy for class {} is: ".format(i),accuracy)
    print("There are {} predictions that are not 100 percent confident".format(confid))

ave_accuracy = sum(accuracy_list)/len(accuracy_list)
ave_time = sum(time_ave)/len(time_ave)
print("Total accuracy: ",ave_accuracy)
print("Average time: ",ave_time)