import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def displayImg(img,cmap=None):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Getting the raw images from the directory
path = 'C:/PythonPrograms/GitClones/StarTracker_Thesis/Star Tracker/conv-net_initial-results/stellarium_images/'
dataset = []
for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path,filename))
    if img is not None:
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_img = gray_img[200:500,600:1000]
        dataset.append(new_img)

#Saving the default image to a directory
for index,image in enumerate(dataset):
    path = 'dataset/train/'+str(index)+'/'
    file_name = '0.jpg'
    cv2.imwrite(path+file_name,image)