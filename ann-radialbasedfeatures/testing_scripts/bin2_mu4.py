from keras.models import load_model
from numpy.lib.function_base import average

#MODIFY MODEL
my_model = load_model('testing_scripts/model_bin2_mu4.h5')

from module_dependencies.star_image_generator import Generator

generator = Generator(6)
catalogue = generator.catalogue.to_numpy()
attitudes = catalogue[:30,1:3]
#MODIFY MISSING AND UNEXPECTED STAR
missing_star = 4
unexpected_star = 4

images = []
for i,attitude in enumerate(attitudes):
    ra = attitude[0]
    de = attitude[1]
    image = generator.create_star_image(ra,de,0,missing_star,unexpected_star)
    print("Creating data star {} of {}".format(i,len(attitudes)))
    images.append(image)

import time
import numpy as np

def scaling(features):
    rescaled = []
    max_val = sum(features)
    for feature in features:
        rescaled.append(feature/max_val)
    return rescaled

time_array = []
#MODIFY BIN INCREMENT
for i,image in enumerate(images):
    print("Calculating attitude star {} of {}".format(i,len(images)))
    t0 = time.perf_counter()
    filter = Filter(image)
    filtered_image = filter.filter_image(4)
    features = generator.extract_rb_features(2,image)
    scaled_features = np.array([scaling(features)])
    result = np.argmax(my_model.predict(scaled_features))
    time_elapsed = time.perf_counter() - t0
    time_array.append(time_elapsed)

average_time = sum(time_array)/len(time_array)
print(average_time)