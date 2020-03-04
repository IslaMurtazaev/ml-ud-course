#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

from tools.feature_format import featureFormat

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))


print 'number of data points (people)', len(enron_data.keys())
print 'number of features', len(enron_data[enron_data.keys()[0]].keys())

number_of_poi = 0
for key in enron_data.keys():
    if enron_data[key]['poi']:
        number_of_poi += 1
print 'number of POI', number_of_poi

# print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

# salaries = []
# emails = dict()
# for key in enron_data.keys():
#     salary = enron_data[key]['salary']
#     if salary != 'NaN':
#         salaries.append(salary)
#
#     email = enron_data[key]['email_address']
#     if email != 'NaN':
#         emails[email] = key


print len([1 for key in enron_data.keys() if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi']])
print 125 / (146 / 100)

import pprint

pp = pprint.PrettyPrinter()
pp.pprint(enron_data)
#
# pp.pprint(featureFormat(enron_data, ["poi", "salary", "bonus"]))
