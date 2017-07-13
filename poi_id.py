
# coding: utf-8

# In[30]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[31]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print len(data_dict)
print len(data_dict.values()[0])
data_dict.values()[0].keys()


# In[32]:

poi = 0
for value in data_dict.values():
    if value['poi'] == True:
        poi += 1
print poi


# In[33]:

#Visulize the outliers and remove the outlier which it should be.

import matplotlib.pyplot
get_ipython().magic(u'matplotlib inline')
for value in data_dict.values():
    salary = value['salary']
    bonus = value['bonus']
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

data_dict.pop( 'TOTAL', 0 )
for value in data_dict.values():
    salary = value['salary']
    bonus = value['bonus']
    matplotlib.pyplot.scatter( salary, bonus )


# In[34]:

###the fraction of valid data points (exclude NaN values) was calculated for each feature 
###across three classes: everyone in the dataset, non-POIs in the dataset, and POIs in the dataset. 


import math
from collections import defaultdict

dic1= {}
fraction = defaultdict(list)

for item in data_dict.values()[0]:
    num = 0
    for value in data_dict.values():
        if value[item] != 'NaN':
            num += 1
            dic1[item]=num


for k, v in dic1.items():
    fra_all = round(v/145.0, 2)
    fraction[k].append(fra_all)
   

    
dic2= {}    
for item in data_dict.values()[0]:
    num = 0
    for value in data_dict.values():
        if value[item] != 'NaN' and value['poi'] == 0:
            num += 1
            dic2[item] = num

for k, v in dic2.items():
    fra_non_POI = round(v/127.0, 2)
    fraction[k].append(fra_non_POI)
   


    
dic3= {}
for item in data_dict.values()[0]:
    num = 0
    for value in data_dict.values():
        if value[item] != 'NaN' and value['poi'] == 1:
            num += 1
            dic3[item] = num  

for k, v in dic3.items():
    fra_POI = round(v/18.0, 2)
    fraction[k].append(fra_POI)
    
print fraction

###For the financial features, the total_payments, total_stock_value, expenses, have 100% coverage\
###for POIs and non 100% coverage for non-POIs. This could lead to the classifier simply using the \
###existence (or non-existence) of those features on a testing point to predict if it was a POI, \
###instead of using the values. So those features were discarded.
###In addition, restricted_stock_deferred, and director_fees all had no POI data coverage,\
###so they were discarded as the same reason as above.
###For the email features, the email_address feature had 100% POI coverage, so it was also discarded.


# In[35]:

# Add two new features in Enron data.

new_features = ['fraction_message_with_poi', 'fraction_bonus' ]

for k, v in data_dict.items():
    if v['from_messages'] != 'NaN' and v['to_messages'] !='NaN' and     v['from_poi_to_this_person']!='NaN' and v['from_this_person_to_poi']!='NaN':
        
        all_messages = v['from_messages'] + v['to_messages']
        all_messages_with_poi =  v['from_poi_to_this_person'] + v['from_this_person_to_poi']
        fraction_message_with_poi = round(float(all_messages_with_poi)/float(all_messages), 2)
    else:
        fraction_message_with_poi = 'NaN'
        
    data_dict[k]['fraction_message_with_poi'] = fraction_message_with_poi
    
    if v['bonus'] != 'NaN' and v['total_payments'] !='NaN':
        
        fraction_bonus = round(float(v['bonus'])/float(v['total_payments']), 2)
    else:
        fraction_bonus = 'NaN'
    data_dict[k]['fraction_bonus'] = fraction_bonus



# In[36]:

#Store the new features to my_dataset.
my_dataset = data_dict


# In[37]:

#Using the feature scores from the SelectKBest to do feature selection, and store the final features in my_features.
features_list = ['poi','salary','to_messages', 'deferral_payments',                  'deferred_income', 'long_term_incentive', 'shared_receipt_with_poi', 'loan_advances',                 'from_messages', 'bonus',  'from_poi_to_this_person', 'from_this_person_to_poi',                 'restricted_stock', 'exercised_stock_options']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k='all')
selector.fit(features, labels)
scores = selector.scores_
feature_scores = zip(features_list[1:],scores)
feature_scores = sorted(feature_scores, key=lambda x:x[1], reverse=True)
print feature_scores



features_list_new = ['poi','salary','to_messages', 'deferral_payments', 'fraction_message_with_poi',                  'deferred_income', 'long_term_incentive', 'shared_receipt_with_poi', 'loan_advances',                 'from_messages', 'bonus',  'from_poi_to_this_person', 'from_this_person_to_poi',                 'restricted_stock', 'exercised_stock_options', 'fraction_bonus']
data = featureFormat(my_dataset, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k='all')
selector.fit(features, labels)
scores = selector.scores_
feature_scores = zip(features_list_new[1:],scores)
feature_scores = sorted(feature_scores, key=lambda x:x[1], reverse=True)
print feature_scores


# In[47]:

### try GNB with PCA
from tester import test_classifier, dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


estimators = [('scaler' , MinMaxScaler()),('selector', SelectKBest(f_classif)),('pca', PCA()),('clf', GaussianNB())]
pipeline=Pipeline(estimators)
print pipeline

parameters = {'selector__k':[4,5,8,10,13],'pca__n_components':[2,3,4]}
clf = GridSearchCV(pipeline, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_



# In[40]:

###  try GNB without PCA
scaler = MinMaxScaler()

estimators = [('scaler' , MinMaxScaler()),('selector', SelectKBest(f_classif)),('clf', GaussianNB())]
pipeline=Pipeline(estimators)

parameters = {'selector__k':[4, 5, 8, 10, 13]}
clf = GridSearchCV(pipeline, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_


# In[41]:

### try DT with PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

estimators = [('pca', PCA()), ('clf', DecisionTreeClassifier())]
pipeline=Pipeline(estimators)
print pipeline

parameters = {'pca__n_components':[2, 4, 6, 8, 10],'clf__min_samples_split':[2,3,4,5,8]}
clf = GridSearchCV(pipeline, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_


# In[44]:

### try DT without PCA

DT = DecisionTreeClassifier()
parameters = {'min_samples_split':[2,3,4,5,8]}
clf = GridSearchCV(DT, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_


# In[49]:

### try KNeighbors with PCA
from sklearn.neighbors import KNeighborsClassifier

estimators = [('selector', SelectKBest(f_classif)), ('pca', PCA()), ('KN', KNeighborsClassifier())]
pipeline = Pipeline(estimators)
print pipeline

parameters = {'selector__k':[4,5,8,10,13],'pca__n_components':[2,3,4], 'KN__n_neighbors': [2,3,5,8]}
clf = GridSearchCV(pipeline, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_


# In[51]:

### try KNeighbors without PCA
estimators = [('selector', SelectKBest(f_classif)), ('KN', KNeighborsClassifier())]
pipeline = Pipeline(estimators)
print pipeline

clf = KNeighborsClassifier()
parameters = {'selector__k':[4,5,8,10,13],'KN__n_neighbors': [2,3,5,8]}
clf = GridSearchCV(pipeline, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_


# In[52]:

### try AdaBoostClassifier with PCA
from sklearn.ensemble import AdaBoostClassifier
estimators = [('pca', PCA(n_components=8)), ('clf', AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=3)))]
pipeline = Pipeline(estimators)
print pipeline


parameters = {'clf__n_estimators':[5,10,30,50], 'clf__learning_rate':[0.1, 0.4,0.6]}
clf = GridSearchCV(pipeline, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_


# In[53]:

### try AdaBoostClassifier without PCA
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=3))

parameters = {'n_estimators':[5,10,30,50], 'learning_rate':[0.1,0.4,0.6]}
clf = GridSearchCV(clf, parameters)
test_classifier(clf, my_dataset, features_list_new)
print 'best_params:', clf.best_params_


# In[55]:

### After running different algorithms, GaussianNB without PCA has relatively best output, 
### so I will choose it as final algorithm.

scaler = MinMaxScaler()
estimators = [('scaler' , MinMaxScaler()),('selector', SelectKBest(f_classif, k=4)),('clf', GaussianNB())]
my_clf = Pipeline(estimators) 
test_classifier(my_clf, my_dataset, my_features_list)

scores = selector.scores_
feature_scores = zip(features_list_new[1:],scores)
feature_scores = sorted(feature_scores, key=lambda x:x[1], reverse=True)
print feature_scores


# In[57]:

features, scores = zip(*feature_scores)

my_features_list = ['poi'] + list(features[:4])
print my_features_list


# In[60]:

### Dump my classifier, dataset, and features_list.

dump_classifier_and_data(my_clf, my_dataset, my_features_list)



# In[ ]:



