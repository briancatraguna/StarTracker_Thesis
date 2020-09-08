#So this program is used to build a classifier of different clothing types with Keras and Convolutional Neural Networks.
#The data is imported from the built-in fashion MNIST dataset from the package included in Keras.
#The dataset consists of 10 different clothing types with 28 x 28 grayscale images.

#Training set -> 60,000 images
#Test set -> 10,000 images

# Label	Description
# 0	    T-shirt/top
# 1	    Trouser
# 2	    Pullover
# 3	    Dress
# 4	    Coat
# 5	    Sandal
# 6	    Shirt
# 7	    Sneaker
# 8	    Bag
# 9	    Ankle boot

#Important modules
import numpy as np
import matplotlib.pyplot as plt

#Importing the data set
from keras.datasets import fashion_mnist
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

#Normalize the pixels
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()

#Include a fourth dimension of the channel, making it a tensor to be fed in the model
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#Convert the y_train and y_test to be one-hot encoded because they're not a regression problem, to do categorical analysis by Keras.
from keras.utils import to_categorical
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

#BUILDING THE MODEL
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
print(model.summary())

#TRAINING THE MODEL
model.fit(x_train,y_cat_train,epochs=25)

#SAVING THE MODEL
model.save('C:\PythonPrograms\Deep Learning Models\FashionMNISTTwentyFiveEpochs.h5')