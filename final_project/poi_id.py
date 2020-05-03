#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier

features_list = ['poi', 'bonus', 'expenses', 'deferred_income']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Remove outliers
my_dataset = data_dict
my_dataset.pop('TOTAL')

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

clf = GaussianNB()

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

test_classifier(clf, my_dataset, features_list)
