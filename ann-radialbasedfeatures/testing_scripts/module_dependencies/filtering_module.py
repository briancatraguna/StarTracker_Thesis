import pandas as pd
import numpy as np
import cv2

class Filter():

    def __init__(self,image) -> None:
        """[filter constructor]

        Args:
            image ([numpy array]): [star image]
        """
        self.image = image

    def filter_image(self,magnitude):
        """[Filters star image. Magnitude M with higher than given will be filtered]

        Args:
            magnitude ([int]): [The magnitude you choose that will be filtered]
        """
        star_image = self.image
        pixel = abs(magnitude-7)
        max_pixel = int(round((pixel/9)*(155)+100))
        t,result = cv2.threshold(star_image,max_pixel,255,cv2.THRESH_TOZERO)
        return result