from keras.models import load_model
import pandas as pd

my_model = load_model('bin1/miss_unexpected_0/model.h5')
testData = pd.read_csv('bin1/miss_unexpected_0/test.csv').to_numpy()
x_test = testData[:,2:]
x_test = x_test/x_test.max()
y_test = testData[:,1]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_test.reshape(-1,1))
y_cat_test = enc.transform(y_test.reshape(-1,1)).toarray()

from sklearn.metrics import classification_report
predictions = my_model.predict_classes(x_test) #The predictions are not in one-hot encoded

predictions = list(predictions)

new_y_test = list(y_test)
counter = -1
comparator = -1
simpan = []
for element in new_y_test:
    if (element!=comparator):
        comparator = element
        counter += 1
    simpan.append(counter)

right = 0
wrong = 0
for i in range(len(simpan)):
    prediction = predictions[i]
    actual = simpan[i]
    if (prediction==actual):
        right += 1
    else:
        wrong += 1
        
accuracy = right/(right+wrong)
print("Accuracy: {}".format(accuracy))