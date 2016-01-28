__author__ = "Chris Lucian"
# numpy is a data shaping and loading module
import numpy as np
# sklearn is a library of machne learning algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import metrics
from sklearn.svm import SVC

# Load the data from the CSV
data = np.genfromtxt('iris.txt', delimiter=',')
# shuffle Data
np.random.shuffle(data)

print("data shape", data.shape)

# load data
# slice the data into taining and test
# Training set is 66% of the data
train = np.vstack((data[0::3, :], data[1::3, :]))
# Test set is 33% of the data
test = data[2::3, :]

# uncomment to shuffle train and test sets
# np.random.shuffle(train)
# np.random.shuffle(test)

print(train, test)
print("train and test shapes", train.shape, test.shape)

# slice to separate inputs and outputs for train and test sets

train_input = train[:, 0:4]
train_output = train[:, 4]

print("train in and out", train_input.shape, train_output.shape)

test_input = test[:, 0:4]
test_output = test[:, 4]

print("test in and out", test_input.shape, test_output.shape)

# train an SVC (Suppot Vector Classifier)

# create the classifier
classifier = RandomForestClassifier()  # or SVC()
# learn the data
classifier.fit(train_input, train_output)
print(test_output)
# predict the output of the test input
predicted = classifier.predict(test_input)

print(predicted)

# Calculate the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_output, predicted, pos_label=2)
# Calculate the area under the ROC curve
auc = metrics.auc(fpr, tpr)

print(auc)

# predict the test set values
# get our AUC and accuracy
