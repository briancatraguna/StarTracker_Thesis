import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

#Defining functions to make life easier

#Displaying images and setting colormap
def displayImg(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    plt.show()

#Changing integer labels to actual names
def index_to_label (prediction):
    prediction = prediction.reshape(1)
    int_prediction = int(prediction)
    if int_prediction == 0:
        label = 'T-Shirt'
    elif int_prediction == 1:
        label = 'Trouser'
    elif int_prediction == 2:
        label = 'Pullover'
    elif int_prediction == 3:
        label = 'Dress'
    elif int_prediction == 4:
        label = 'Coat'
    elif int_prediction == 5:
        label = 'Sandal'
    elif int_prediction == 6:
        label = 'Shirt'
    elif int_prediction == 7:
        label = 'Sneaker'
    elif int_prediction == 8:
        label = 'Bag'
    elif int_prediction == 9:
        label = 'Ankle Boot'
    else:
        label = 'None'
    return label

#Frame formatting for classification in the backend (Note: This is according to my webcam resolution, need to change if webcam res is different)
def frame_formatting(frame):
    #Numpy slicing
    frame = np.delete(frame,[range(480,640)],axis=1)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame,(28,28),interpolation=cv2.INTER_AREA)
    resized = resized.reshape(1,28,28,1)
    return resized

#Importing the MNIST data set
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

#Loading the model
from keras.models import load_model
model = load_model('C:\PythonPrograms\Deep Learning Models\FashionMNISTTwentyFiveEpochs.h5')

#Feed in the test data
print(model.metrics_names)
model.evaluate(x_test,y_cat_test)

#Model stats
from sklearn.metrics import classification_report
predictions = model.predict_classes(x_test) #The predictions are not in one-hot encoded
print(classification_report(y_test,predictions))

# Using the webcam
while(True):
    ret,frame = cap.read()
    backend = frame_formatting(frame)
    prediction = model.predict_classes(backend)
    label = index_to_label(prediction)
    preview = 'Detected Object: ' + label
    cv2.putText(frame,preview,org=(10,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()