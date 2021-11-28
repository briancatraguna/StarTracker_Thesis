from numpy.core.fromnumeric import argmax
from feature_detector import multitriangles_detector,net_feature,displayImg,centroiding
from create_star_module import create_star_image
import cv2
import numpy as np
from math import sqrt

#Net Algorithm
def net_feature(image,distance_to_center_filter_pixels,n):
    img = image
    
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

    #Detect stars
    keypoints = detector.detect(img)
    coord = []
    for index,keypoint in enumerate(keypoints):
        x_centralstar = int(round(keypoints[index].pt[0]))
        y_centralstar = int(round(keypoints[index].pt[1]))
        distance_to_center = sqrt(((x_centralstar-x)**2)+((y_centralstar-y)**2))
        coord.append([x_centralstar,y_centralstar,distance_to_center])
        # cv2.circle(img,center=(x_centralstar,y_centralstar),radius=2,color=(255,0,0),thickness=2)

    #Find stars within radius
    stars_within_radius = []
    for coordinate in coord:
        dist_to_center = coordinate[2]
        if (dist_to_center<distance_to_center_filter_pixels):
            stars_within_radius.append(coordinate)
    if len(stars_within_radius)==0:
        return img

    #Find magnitude of star within radius
    magnitude_of_star_within_radius = []
    for star in stars_within_radius:
        x = star[0]
        y = star[1]
        brightness_value = img[y,x][0]
        magnitude_of_star_within_radius.append(brightness_value)

    #Find index of brightest star and coordinate
    max_index = argmax(magnitude_of_star_within_radius)
    pivot_coord = stars_within_radius[max_index]
    cv2.circle(img,center=(pivot_coord[0],pivot_coord[1]),radius=2,color=(255,0,0),thickness=2)

    #Find nearest stars
    x_pivot = pivot_coord[0]
    y_pivot = pivot_coord[1]
    distances = []
    for coordinate in coord:
        x_relate = coordinate[0]
        y_relate = coordinate[1]
        delta_x = abs(x_pivot - x_relate)
        delta_y = abs(y_pivot - y_relate)
        distance = sqrt((delta_x*delta_x)+(delta_y*delta_y))
        if (distance == 0):
            continue
        distances.append([distance,coordinate])
    
    distances = sorted(distances ,key=lambda row:row[0])
    neighbor_stars = []
    for distance in distances:
        neighbor_stars.append(distance[1])

    #Draw circle and line
    for i,star in enumerate(neighbor_stars):
        cv2.circle(img,center=(star[0],star[1]),radius=2,color=(255,0,0),thickness=2)
        cv2.line(img,(pivot_coord[0],pivot_coord[1]),(star[0],star[1]),(255,0,0),2)
        if i == n-1:
            break

    return img

# img = cv2.imread('test.jpg')
# pixel_from_center_filter = 100
# net_feature_img = net_feature(img,pixel_from_center_filter,4)
# displayImg(net_feature_img)

#Net Algorithm
def multitriangle(image,distance_to_center_filter_pixels,n):
    img = image
    
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

    #Detect stars
    keypoints = detector.detect(img)
    coord = []
    for index,keypoint in enumerate(keypoints):
        x_centralstar = int(round(keypoints[index].pt[0]))
        y_centralstar = int(round(keypoints[index].pt[1]))
        distance_to_center = sqrt(((x_centralstar-x)**2)+((y_centralstar-y)**2))
        coord.append([x_centralstar,y_centralstar,distance_to_center])
        # cv2.circle(img,center=(x_centralstar,y_centralstar),radius=2,color=(255,0,0),thickness=2)

    #Find stars within radius
    stars_within_radius = []
    for coordinate in coord:
        dist_to_center = coordinate[2]
        if (dist_to_center<distance_to_center_filter_pixels):
            stars_within_radius.append(coordinate)
    if len(stars_within_radius)==0:
        return img

    #Find magnitude of star within radius
    magnitude_of_star_within_radius = []
    for star in stars_within_radius:
        x = star[0]
        y = star[1]
        brightness_value = img[y,x][0]
        magnitude_of_star_within_radius.append(brightness_value)

    #Find index of brightest star and coordinate
    max_index = argmax(magnitude_of_star_within_radius)
    pivot_coord = stars_within_radius[max_index]
    cv2.circle(img,center=(pivot_coord[0],pivot_coord[1]),radius=2,color=(255,0,0),thickness=2)

    #Find nearest stars
    x_pivot = pivot_coord[0]
    y_pivot = pivot_coord[1]
    distances = []
    for coordinate in coord:
        x_relate = coordinate[0]
        y_relate = coordinate[1]
        delta_x = abs(x_pivot - x_relate)
        delta_y = abs(y_pivot - y_relate)
        distance = sqrt((delta_x*delta_x)+(delta_y*delta_y))
        if (distance == 0):
            continue
        distances.append([distance,coordinate])
    
    distances = sorted(distances ,key=lambda row:row[0])
    neighbor_stars = []
    for distance in distances:
        neighbor_stars.append(distance[1])

    #Draw circle and line
    for i,star in enumerate(neighbor_stars):
        cv2.circle(img,center=(star[0],star[1]),radius=2,color=(255,0,0),thickness=2)
        cv2.line(img,(pivot_coord[0],pivot_coord[1]),(star[0],star[1]),(255,0,0),2)
        for i_2,star_2 in enumerate(neighbor_stars):
            cv2.line(img,(star[0],star[1]),(star_2[0],star_2[1]),(255,0,0),2)
            if i_2 == n-1:
                break
        if i == n-1:
            break

    return img

# img = cv2.imread('test.jpg')
# pixel_from_center_filter = 100
# net_feature_img = multitriangle(img,pixel_from_center_filter,4)
# displayImg(net_feature_img)