#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from math import isnan

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
 'other',
 'long_term_incentive',
 'expenses',
 'deferral_payments',
 'restricted_stock_deferred',
 'deferred_income',
 'from_poi_ratio',
 'to_poi_ratio'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl",
 "rb") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for 
### easy export below.
my_dataset = data_dict


### Task 2: Remove outliers
my_dataset.pop("TOTAL", None)
### Task 3: Create new feature(s)

for key, value in my_dataset.items():
    if value["from_messages"] != 0:
        value["to_poi_ratio"] = float(value['from_this_person_to_poi'])/float(value["from_messages"])
        if isnan(value["to_poi_ratio"]):
            value["to_poi_ratio"] = "NaN"
    else:
        value["to_poi_ratio"] = "NaN"
        
    if value["to_messages"] != 0:
        value["from_poi_ratio"] = float(value['from_poi_to_this_person'])/float(value["to_messages"])
        if isnan(value["from_poi_ratio"]):
            value["from_poi_ratio"] = "NaN"
    else:
        value["from_poi_ratio"] = "NaN"

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, \
 sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=1)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
etc = DecisionTreeClassifier()

results = {}

param_grid = {"random_state": [1], 'criterion':["entropy", "gini"], \
             'min_samples_split': [10,20,30,50], "max_features": ["auto", "sqrt", "log2"], \
             'max_depth': [10,20,30,50]}

from sklearn import grid_search
dtr = grid_search.GridSearchCV(etc, param_grid, scoring = "precision", cv=5)
dtr.fit(features_train, labels_train)

print(dtr.best_params_)

clf = dtr.best_estimator_
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print("Results on the test set:")
print("accuracy :", round(accuracy_score(pred, labels_test),3))
print("precision :", round(precision_score(labels_test, pred), 3))
print("recall :", round(recall_score(labels_test, pred), 3))


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)