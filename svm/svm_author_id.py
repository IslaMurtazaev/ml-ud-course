#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train) / 100]
# labels_train = labels_train[:len(labels_train) / 100]

from sklearn.svm import SVC

clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
svc_prediction = clf.predict(features_test)


from sklearn.metrics import accuracy_score

print 'Accuracy: ' + str(accuracy_score(svc_prediction, labels_test))


print 'Number of Chris\' emails ' + str(len(filter(lambda x: x == 1, svc_prediction)))
