#!/usr/bin/python

import pickle
import sys
from matplotlib import pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


# remove outlier
data = sorted(data, key=lambda x: x[0])[:len(data)-1]


import numpy
numpy_data = numpy.array(data)
salaries = numpy_data[:, [0]]
bonuses = numpy_data[:, [1]]

for row in data:
    salary = row[0]
    bonus = row[1]
    plt.scatter(salary, bonus)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(salaries, bonuses)
bonuses_pred = reg.predict(salaries)

plt.xlabel('salary')
plt.ylabel('bonus')
plt.plot(salaries, bonuses_pred, color='red')
plt.show()

