#Importing the necessary libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def displayImg(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Read image
img = cv2.imread("dataset/train/0/0.jpg")
ret,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)

#Find the center of the image
height,width,col = img.shape
coordinate = [height/2,width/2]
y = int(coordinate[0])
x = int(coordinate[1])

#Set up the detector
params = cv2.SimpleBlobDetector_Params()
params.filterByInertia = False
params.filterByConvexity = False
params.minThreshold = 120
params.maxThreshold = 200
params.filterByColor = True
params.blobColor = 255
params.filterByArea = False
params.minArea = 10000
detector = cv2.SimpleBlobDetector_create(params)

#Find the pivot point (Star closest to the center)
i = 2
stars_coordinate = []
while True:
    croppedimg = img[y-i:y+i,x-i:x+i,:]
    y_crop,x_crop,col = croppedimg.shape
    keypoints = detector.detect(croppedimg)
    if len(keypoints) == 3:
        for index,keypoint in enumerate(keypoints):
            x_centralstar_crop = int(keypoints[index].pt[0])
            y_centralstar_crop = int(keypoints[index].pt[1])
            coord_x_centralstar = int(x_centralstar_crop + (width-x_crop)/2)
            coord_y_centralstar = int(y_centralstar_crop + (height-y_crop)/2)
            stars_coordinate.append([coord_x_centralstar,coord_y_centralstar])
            print("COORDINATES: \n","x: ",coord_x_centralstar,"\n","y: ",coord_y_centralstar)
        break
    i+=2

#Extracting coordinates of center star
for coord in stars_coordinate:
    center = (coord[0],coord[1])
    cv2.circle(img,center,2,(255,0,0),2)

displayImg(img)