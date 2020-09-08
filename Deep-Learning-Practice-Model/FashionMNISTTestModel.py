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

#Let's try to predict
from sklearn.metrics import classification_report
predictions = model.predict_classes(x_test) #The predictions are not in one-hot encoded
print(classification_report(y_test,predictions))