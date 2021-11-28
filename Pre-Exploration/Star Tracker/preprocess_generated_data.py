from brightest_star_module import net_feature,multitriangle
import os
import cv2

MULTITRIANGLES = True
DISTANCE_TO_CENTER_IN_PIXELS = 300
NUMBER_OF_NEIGHBORING_STARS = 4

####################### UNCHANGE #########################
root_preprocessed_folder = 'preprocessed_dataset/'
dataset_folder = 'dataset/'

for i,filename in enumerate(os.listdir(dataset_folder)):

    read_image = cv2.imread(dataset_folder+filename)
    if MULTITRIANGLES:
        read_image = multitriangle(
            read_image,
            DISTANCE_TO_CENTER_IN_PIXELS,
            NUMBER_OF_NEIGHBORING_STARS
            )
        cv2.imwrite(root_preprocessed_folder+filename,read_image)
    else:
        read_image = net_feature(
            read_image,
            DISTANCE_TO_CENTER_IN_PIXELS,
            NUMBER_OF_NEIGHBORING_STARS
        )
    cv2.imwrite(root_preprocessed_folder+filename,read_image)