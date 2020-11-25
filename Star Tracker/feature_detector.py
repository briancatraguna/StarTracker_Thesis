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


#Net Algorithm
def net_feature(path):
    img = cv2.imread(path)   
    print("ORIGINAL SHAPE: ",img.shape)

    #Find the center of the image
    height,width,col = img.shape
    coordinate = [height/2,width/2]
    y = int(coordinate[0])
    x = int(coordinate[1])

    #Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = 50
    params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.minArea = 1
    detector = cv2.SimpleBlobDetector_create(params)

    #Find the pivot point (Star closest to the center)
    i = 2
    stars_coordinate = []
    while True:
        top = y-i
        down = y+i
        vertical = abs(top-down)
        if vertical >= height:
            new_i = height/2
            top = int(round(y-new_i))
            down = int(round(y+new_i))
        left = x-i
        right = x+i
        horizontal = abs(left-right)
        if horizontal >= width:
            new_i = width/2
            left = int(round(x-new_i))
            right =int(round(x+new_i))
        croppedimg = img[top:down,left:right,:]
        y_crop,x_crop,col = croppedimg.shape
        keypoints = detector.detect(croppedimg)
        print(len(keypoints))
        if len(keypoints) > 3:
            print("Cropped size: \nx: {}\ny: {}".format(x_crop,y_crop))
            print("Full size: \nx: {}\ny: {}".format(width,height))
            for index,keypoint in enumerate(keypoints):
                x_centralstar_crop = int(round(keypoints[index].pt[0]))
                y_centralstar_crop = int(round(keypoints[index].pt[1]))
                print("Crop Coordinates: \nx: {}\ny: {}".format(x_centralstar_crop,y_centralstar_crop))
                coord_x_centralstar = int(round(x_centralstar_crop + ((width-x_crop)/2)))
                coord_y_centralstar = int(round(y_centralstar_crop + ((height-y_crop)/2)))
                stars_coordinate.append([coord_x_centralstar,coord_y_centralstar])
                print("COORDINATES: \n","x: ",coord_x_centralstar,"\n","y: ",coord_y_centralstar)
            break
        i+=2

    #Draw circles on important stars and find the pivot star
    dist_to_center = []
    for coord in stars_coordinate:
        center = (coord[0],coord[1])
        dist_x_to_center = abs(coord[0] - x)
        dist_y_to_center = abs(coord[1] - y)
        resultant = (dist_x_to_center**2+dist_y_to_center**2)**1/2
        dist_to_center.append(resultant)
        cv2.circle(img,center,2,(255,0,0),2)
    
    ind = dist_to_center.index(min(dist_to_center))
    pivot_star_coord = tuple(stars_coordinate[ind])
    del stars_coordinate[ind]

    #Draw lines from pivot point to other stars
    for coord in stars_coordinate:
        cv2.line(img,pivot_star_coord,tuple(coord),(255,0,0),2)

    return img

#Multitriangles Algorithm
def multitriangles_detector(path):
    img = cv2.imread(path)  

    #Find the center of the image
    height,width,col = img.shape
    coordinate = [height/2,width/2]
    y = int(coordinate[0])
    x = int(coordinate[1])

    #Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = 50
    params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.minArea = 1
    detector = cv2.SimpleBlobDetector_create(params)

    #Find the pivot point (Star closest to the center)
    i = 2
    stars_coordinate = []
    while True:
        top = y-i
        down = y+i
        vertical = abs(top-down)
        if vertical >= height:
            new_i = height/2
            top = int(round(y-new_i))
            down = int(round(y+new_i))
        left = x-i
        right = x+i
        horizontal = abs(left-right)
        if horizontal >= width:
            new_i = width/2
            left = int(round(x-new_i))
            right =int(round(x+new_i))
        croppedimg = img[top:down,left:right,:]
        y_crop,x_crop,col = croppedimg.shape
        keypoints = detector.detect(croppedimg)
        print(len(keypoints))
        if len(keypoints) > 3:
            print("Cropped size: \nx: {}\ny: {}".format(x_crop,y_crop))
            print("Full size: \nx: {}\ny: {}".format(width,height))
            for index,keypoint in enumerate(keypoints):
                x_centralstar_crop = int(round(keypoints[index].pt[0]))
                y_centralstar_crop = int(round(keypoints[index].pt[1]))
                print("Crop Coordinates: \nx: {}\ny: {}".format(x_centralstar_crop,y_centralstar_crop))
                coord_x_centralstar = int(round(x_centralstar_crop + ((width-x_crop)/2)))
                coord_y_centralstar = int(round(y_centralstar_crop + ((height-y_crop)/2)))
                stars_coordinate.append([coord_x_centralstar,coord_y_centralstar])
                print("COORDINATES: \n","x: ",coord_x_centralstar,"\n","y: ",coord_y_centralstar)
            break
        i+=2

    for coord in stars_coordinate:
        coord = tuple(coord)
        cv2.circle(img,coord,2,(255,0,0),-1)
        for other_coord in stars_coordinate:
            if other_coord == coord:
                continue
            else:
                other_coord = tuple(other_coord)
                cv2.line(img,coord,other_coord,(255,0,0),2)

    return img

#Circle
def centroiding(path):
    img = cv2.imread(path)   
    print("ORIGINAL SHAPE: ",img.shape)

    #Find the center of the image
    height,width,col = img.shape
    coordinate = [height/2,width/2]
    y = int(coordinate[0])
    x = int(coordinate[1])

    #Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minThreshold = 50
    params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False
    params.minArea = 1
    detector = cv2.SimpleBlobDetector_create(params)

    #Find the pivot point (Star closest to the center)
    i = 2
    stars_coordinate = []
    while True:
        top = y-i
        down = y+i
        vertical = abs(top-down)
        if vertical >= height:
            new_i = height/2
            top = int(round(y-new_i))
            down = int(round(y+new_i))
        left = x-i
        right = x+i
        horizontal = abs(left-right)
        if horizontal >= width:
            new_i = width/2
            left = int(round(x-new_i))
            right =int(round(x+new_i))
        croppedimg = img[top:down,left:right,:]
        y_crop,x_crop,col = croppedimg.shape
        keypoints = detector.detect(croppedimg)
        print(len(keypoints))
        if len(keypoints) > 3:
            print("Cropped size: \nx: {}\ny: {}".format(x_crop,y_crop))
            print("Full size: \nx: {}\ny: {}".format(width,height))
            for index,keypoint in enumerate(keypoints):
                x_centralstar_crop = int(round(keypoints[index].pt[0]))
                y_centralstar_crop = int(round(keypoints[index].pt[1]))
                print("Crop Coordinates: \nx: {}\ny: {}".format(x_centralstar_crop,y_centralstar_crop))
                coord_x_centralstar = int(round(x_centralstar_crop + ((width-x_crop)/2)))
                coord_y_centralstar = int(round(y_centralstar_crop + ((height-y_crop)/2)))
                stars_coordinate.append([coord_x_centralstar,coord_y_centralstar])
                print("COORDINATES: \n","x: ",coord_x_centralstar,"\n","y: ",coord_y_centralstar)
            break
        i+=2

    #Draw circles on important stars and find the pivot star
    dist_to_center = []
    for coord in stars_coordinate:
        center = (coord[0],coord[1])
        dist_x_to_center = abs(coord[0] - x)
        dist_y_to_center = abs(coord[1] - y)
        resultant = (dist_x_to_center**2+dist_y_to_center**2)**1/2
        dist_to_center.append(resultant)
        cv2.circle(img,center,2,(255,0,0),2)
    
    return img

import time

path = 'dataset/train/0/0.jpg'
img = cv2.imread(path)
displayImg(img)
img_centroid = centroiding(path)
displayImg(img_centroid)

start_time = time.time()
img_net = net_feature(path)
final_time = time.time()
net_time = start_time - final_time
displayImg(img_net)

start_time = time.time()
img_multitriangle = multitriangles_detector(path)
final_time = time.time()
multitriangle_time = final_time-start_time
displayImg(img_multitriangle)

print("Time to do net algorithm:",net_time)
print("Time to do multitirangle algorithm:",multitriangle_time)