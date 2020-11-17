#Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def displayImg(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Read image
im = cv2.imread("dataset/train/0/0.jpg",0)
displayImg(im)

#Set up the detector with default parameters
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

#Detect blobs
keypoints = detector.detect(im)

#Draw detected blobs as red circles
#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im,keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Show keypoints
displayImg(im_with_keypoints)