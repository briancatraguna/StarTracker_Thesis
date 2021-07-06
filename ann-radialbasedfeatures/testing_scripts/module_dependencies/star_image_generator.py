import pandas as pd
import nested_function as nf
import numpy as np
from math import degrees,sqrt,atan
import cv2

class Generator():

    def __init__(self) -> None:
        self.col_list = ["Star ID","RA","DE","Magnitude"]
        self.catalogue = pd.read_csv('module_dependencies/Below_4.0_SAO.csv',usecols=self.col_list)

    #Extract features function
    def extract_rb_features(self,bin_increment,image):
        """[This function extracts the radial basis features from a given star image and returns the bin feature vectors]

        Args:
            bin_increment ([int]): [The bin increment is the delta theta for the histogram of features (IN DEGREES)]
            image ([numpy array]): [The star image]
            myu ([float]): [length per pixel]
            f ([float]): [focal length]
        """
        myu = 1.12*(10**-6)
        f = 0.00304
        
        #Defining some reusable variables to use
        half_length_pixel = image.shape[1]/2
        half_width_pixel = image.shape[0]/2
        FOVy_half = degrees(atan((half_width_pixel*myu)/f))

        #Initializing the bin list
        length_of_bin = FOVy_half//bin_increment
        bin_list = [0] * int(length_of_bin)

        #Get all the centroids
        image = image.astype('uint8')
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

        keypoints = detector.detect(image)
        #Iterating through all the stars present
        for index,keypoint in enumerate(keypoints):
            x_centralstar = int(round(keypoints[index].pt[0]))
            y_centralstar = int(round(keypoints[index].pt[1]))
            #Converting to origin-in-the-middle coordinates
            x = x_centralstar - half_length_pixel
            y = half_width_pixel - y_centralstar
            pixel_distance_to_center = sqrt((x**2)+(y**2))
            angular_distance_to_center = round(degrees(atan((pixel_distance_to_center*myu)/f)),3)
            if angular_distance_to_center > FOVy_half:
                continue
            lower_bound = 0
            upper_bound = bin_increment
            bin_index = 0
            #Evaluate which bin is this star in
            while upper_bound <= FOVy_half:
                if lower_bound <= angular_distance_to_center < upper_bound:
                    bin_list[bin_index] += 1
                lower_bound += bin_increment
                upper_bound += bin_increment
                bin_index += 1

        return bin_list

    #Create star image
    def create_star_image(self,ra,de,roll,missing_star,unexpected_star):
        image = nf.create_star_image(ra,de,roll,self.catalogue,missing_star,unexpected_star,0.2)
        return image

generator = Generator()
image = generator.create_star_image(0,0,0,0,0)
features = generator.extract_rb_features(1,image)
print(features)