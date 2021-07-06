from keras.models import load_model

my_model = load_model('model.h5')

from module_dependencies.star_image_generator import Generator

generator = Generator()