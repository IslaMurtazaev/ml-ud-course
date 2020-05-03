#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =\
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

print 'num of all people in test set', len(labels_test)
print 'num of POIs in test set', sum(labels_test)

prediction = clf.predict(features_test)
print 'prediction', prediction
print 'real values', labels_test

true_positives = 0
for i in range(len(labels_test)):
    if labels_test[i] and prediction[i]:
        true_positives += 1

print 'num of true positives', true_positives


from sklearn.metrics import precision_score, recall_score

print 'precision', precision_score(labels_test, prediction)
print 'recall', recall_score(labels_test, prediction)
print 'accuracy', clf.score(features_test, labels_test)


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_positives = 0
false_positives = 0
false_negatives = 0
for i in range(len(true_labels)):
    if predictions[i]:
        if true_labels[i]:
            true_positives += 1
        else:
            false_positives += 1
    elif true_labels[i]:
        false_negatives += 1


print 'precision formula', true_positives / float(true_positives + false_positives)
print 'recall formula', true_positives / float(true_positives + false_negatives)


