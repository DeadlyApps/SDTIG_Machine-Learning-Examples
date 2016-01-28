from random import shuffle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import metrics
from sklearn.svm import SVC

data = np.genfromtxt('iris.txt', delimiter=',')
np.random.shuffle(data)
print("data shape",data.shape)

#load data
#slice the data into taining and test
train = np.vstack((data[0::3,:], data[1::3,:]))
test = data[2::3,:]

# np.random.shuffle(train)
# np.random.shuffle(test)

print(train, test)
print("train and test shapes", train.shape, test.shape)

#slice class

train_input = train[:,0:4]
train_output = train[:,4]

print("train in and out", train_input.shape, train_output.shape)


test_input = test[:,0:4]
test_output = test[:,4]

print("test in and out", test_input.shape, test_output.shape)

#train an SVC (Suppot Vector Classifier)

classifier = RandomForestClassifier()
classifier.fit(train_input, train_output)
print(test_output)
predicted = classifier.predict(test_input)

print(predicted)

fpr, tpr, thresholds = metrics.roc_curve(test_output, predicted, pos_label=2)
auc = metrics.auc(fpr, tpr)

print(auc)

#predict the test set values
#get our AUC and accuracy