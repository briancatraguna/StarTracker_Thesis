from create_star_module import create_star_image
from random import randint
import cv2

NUMBER_OF_FALSE_STARS = 0
NUMBER_OF_MISSING_STARS = 0
NUMBER_OF_DATA_POINTS = 100
CATALOGUE_PATH = 'Pre-Exploration/Star Tracker/star_catalogue/below_6.0_SAO.csv'

####################### UNCHANGE #########################
saving_directory = 'dataset/'

for i in range(NUMBER_OF_DATA_POINTS):
    
    random_ra = randint(-180,180)
    random_de = randint(-90,90)
    random_roll = randint(0,360)
    star_image = create_star_image(
        random_ra,
        random_de,
        random_roll,
        NUMBER_OF_FALSE_STARS,
        NUMBER_OF_MISSING_STARS,
        CATALOGUE_PATH
        )
    complete_saving_directory = f'{saving_directory}{random_ra}_{random_de}.jpg'
    cv2.imwrite(complete_saving_directory,star_image)
    print("SAVED {} of {} in {}".format(i,NUMBER_OF_DATA_POINTS,complete_saving_directory))
    print(random_ra,random_de)