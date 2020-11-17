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
img = cv2.imread("dataset/train/0/0.jpg",0)
ret,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
displayImg(img)

#Find the center of the image
y,x = img.shape
coordinate = [y/2,x/2]

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
while True:
    croppedimg = img[y-i:y+i,x-i:x+i]
    keypoints = detector.detect(croppedimg)
    if keypoints:
        displayImg(croppedimg)
        x = int(keypoints[0].pt[0])
        y = int(keypoints[0].pt[1])
        break
    i+=2

#Extracting coordinates of center star
keypoint = keypoints[0]
x = int(keypoint.pt[0])
y = int(keypoint.pt[1])
print("COORDINATES: \n","x: ",x,"\n","y: ",y)
center = (x,y)

cv2.circle(img,center,15,[0,0,0],-1)
displayImg(img)