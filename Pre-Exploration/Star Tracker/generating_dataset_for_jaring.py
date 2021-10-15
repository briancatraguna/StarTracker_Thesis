#Importing the necessary libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import imutils
import feature_detector as fd

def displayImg(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

images = []
for i in range(8):
    path = 'dataset_for_multitriangle_algo/train/'+str(i)+'/0.jpg'
    img = cv2.imread(path)
    img = img[:,50:350]
    images.append(img)

for index,image in enumerate(images):
    if index == 0:
        continue
    count = 1
    path = 'dataset_for_multitriangle_algo/train/'+str(index)+'/'
    for angle in np.arange(0,360,0.1):
        rotated = imutils.rotate_bound(image,angle)
        rotated = fd.multitriangles_detector(rotated)
        cv2.imwrite(path+str(count)+'.jpg',rotated)
        print("DONE SAVING...",path+str(count)+'.jpg')
        count+=1