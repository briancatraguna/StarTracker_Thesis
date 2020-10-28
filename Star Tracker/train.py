import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#To prevent overfitting -> training on the same images
#DATA PREPROCESSING
#TRAIN
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    'C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/dataset/train',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

#TEST
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'C:/PythonPrograms/GitClones/CodingProgressforThesis/Star Tracker/dataset/test',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

#BUILDING THE CONVOLUTIONAL NEURAL NETWORK
cnn = tf.keras.models.Sequential() #Sequence of layers
#CONVOLUTION 1
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',input_shape=[64,64,3]))
#POOLING 1
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#CONVOLUTION 2
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
#POOLING 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#CONVOLUTION 3
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
#POOLING 3
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#FLATTENING
cnn.add(tf.keras.layers.Flatten())
#FULL CONNECTION
cnn.add(tf.keras.layers.Dense(units=256,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(8,activation='softmax'))

#TRAINING THE CONVOLUTIONAL NEURAL NETWORK
#Compiling the CNN
cnn.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set,validation_data=test_set,epochs=15)

#SAVING THE MODEL
cnn.save('C:\PythonPrograms\Deep Learning Models\Trained_Mini_StarTracker.h5')

#MAKING A SINGLE PREDICTION
import numpy as np
from keras.preprocessing import image
#First class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/0/1807.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
print(result)
#Second class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/1/603.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(result)
#Third class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/2/1116.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(result)
#Fourth class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/3/1255.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(result)
#Fifth class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/4/2012.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(result)
#Sixth class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/5/863.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(result)
#Seventh class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/6/1794.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(result)
#Eight class
test_image = image.load_img(
    'C:/PythonPrograms\GitClones/CodingProgressforThesis/Star Tracker/dataset/test/7/549.jpg',
    target_size=(64,64)
    )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(result)