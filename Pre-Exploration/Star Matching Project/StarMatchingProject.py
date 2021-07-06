#This is a simple matching program just for my learning tool for my thesis

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#Function for displaying the image
def displayImg(img,cmap=None):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Putting the images in the star catalogue inside a list
path = 'C:/Users/MSI/Desktop/Star Tracker/Project/Random Stars/'
starcatalogue = []
for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path,filename))
    if img is not None:
        starcatalogue.append(img)

#Reading in the cropped file
croppedimg = cv2.imread('C:/Users/MSI/Desktop/Star Tracker/Project/croppedstar.jpg')
secondpic = cv2.imread('C:/Users/MSI/Desktop/Star Tracker/Project/Random Stars/star1.jpg')

#Brute Force Matching with SIFT Descriptors and Ratio Test
#Creating detector object
sift = cv2.xfeatures2d.SIFT_create()
#Find the key points and descriptors off of this object for each image
kpsmall,dscsmall = sift.detectAndCompute(croppedimg,None)
kp = []
dsc = []
for i,img in enumerate(starcatalogue):
    kps,dscs = sift.detectAndCompute(starcatalogue[i],None)
    kp.append(kps)
    dsc.append(dscs)
#Brute force matching
bf = cv2.BFMatcher()
matches = []
for i,descriptor in enumerate(dsc):
    match = bf.knnMatch(dscsmall,dsc[i],k=2)
    matches.append(match)
#Apply ratio test, if distance is small enough, its a good feature to match on
good = []
for i,match in enumerate(matches):
    goodmatch = []
    for match1,match2 in match:
        if match1.distance<0.75*match2.distance:
            goodmatch.append([match1])
    good.append(goodmatch)
#See which match has the most number of good match
lengths = []
for i,match in enumerate(good):
    thelength = len(good[i])
    lengths.append(thelength)
max_index = lengths.index(max(lengths))
#Do the matching of the most matched
sift_matches = cv2.drawMatchesKnn(croppedimg,kpsmall,starcatalogue[max_index],kp[max_index],good[max_index],None,flags=2)
displayImg(sift_matches)
