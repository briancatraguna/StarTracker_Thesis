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
    'dataset_with_features/train',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

#TEST
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'dataset_with_features/test',
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
cnn.save('C:\PythonPrograms\Deep Learning Models\preprocessed_features_model.h5')