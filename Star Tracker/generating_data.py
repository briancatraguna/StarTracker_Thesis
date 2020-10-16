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
path = 'C:/Stellarium_Images/'
dataset = []
for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path,filename))
    if img is not None:
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_img = gray_img[200:500,600:1000]
        dataset.append(new_img)

#Saving the default image to a directory
for index,image in enumerate(dataset):
    path = 'C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/dataset/'+str(index)+'/'
    file_name = str(index)+'_000.jpg'
    cv2.imwrite(path+file_name,image)

test_image = cv2.imread('C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/dataset/0/0_000.jpg')
displayImg(test_image)
#Preprocessing
#Generate random blur
# for i in range(1,10):
#     blur = cv2.blur(test_image,ksize=(i,i))
#     displayImg(blur)

for i in range(1,10):
    kernel = np.ones(shape=(2,2),dtype=np.float32)/i
    destination = cv2.filter2D(test_image,-1,kernel)
    displayImg(destination)
#Generate random noise

#Generate random morphological operations

#Generate random filters
