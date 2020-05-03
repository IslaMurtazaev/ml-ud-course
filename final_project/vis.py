import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL')


features_list = ['poi', 'total_stock_value', 'bonus']

# rescale features

employee_features = []
poi_features = []
for feature in features_list[1:]:
    employee_feature = []
    poi_feature = []
    for name in data_dict.keys():
        if data_dict[name][feature] == 'NaN':
            data_dict[name][feature] = 0.

        if data_dict[name]['poi']:
            poi_feature.append(data_dict[name][feature])
        else:
            employee_feature.append(data_dict[name][feature])

    min_max_feature_scaler = MinMaxScaler()
    min_max_feature_scaler.fit(np.array(employee_feature + poi_feature))
    employee_feature = min_max_feature_scaler.transform(employee_feature)
    poi_feature = min_max_feature_scaler.transform(poi_feature)

    employee_features.append(employee_feature)
    poi_features.append(poi_feature)

    for name in data_dict:
        if data_dict[name][feature] != 'NaN':
            data_dict[name][feature] = min_max_feature_scaler.transform(np.array([data_dict[name][feature]]))

# TODO: Find better features to separate POIs from non-POIs

data = featureFormat(data_dict, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
import pylab as pl

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
# clf = RandomForestClassifier(n_estimators=10, max_depth=4, min_samples_split=3)
clf.fit(features_train, labels_train)  # Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
h = .01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

plt.scatter(employee_features[0], employee_features[1])
plt.scatter(poi_features[0], poi_features[1], c='r')
plt.xlabel(features_list[1])
plt.ylabel(features_list[2])

plt.show()
