#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
t0 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print "pre-processing time:", round(time()-t0, 3), "s"

gnb = GaussianNB()
t0 = time()
author_classifier = gnb.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
authors_classification = author_classifier.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

print
print 'Author prediction accuracy with Gaussian Naive Bayes classifier'
print accuracy_score(authors_classification, labels_test)
