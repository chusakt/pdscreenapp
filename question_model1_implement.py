#%%
# implement for model server: 1 test

import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
# from scipy.stats import kurtosis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import pickle


random_state = 41
# +++++++++++++++++++++++++++++++++++++++++++++++++
# dat_PATH = './DATA/Prepare2/PrepareDualTap_1.csv'
dat_PATH = './DATA/Prepare_fullnof8_C/questionFeaturePDCT.csv'
# dat_PATH = './DATA/Prepare/TremorRest/rest40.csv'
feach = 5     #-- features number each
ft = 20   #-- features number
# +++++++++++++++++++++++++++++++++++++++++++++++++

def load_data(csv_path=dat_PATH):
    return pd.read_csv(csv_path)

data_ = load_data()
data_ = data_.drop('filename', axis=1)
# data_.hist()

#%%
## random stratified with income_cat

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
locc = 'subject' #location of class
for train_index, test_index in split.split(data_, data_[locc]):
    strat_train_set = data_.loc[train_index]
    strat_test_set = data_.loc[test_index]

print('ratio of control to pd')
print('in training set :',strat_train_set[locc].value_counts()[0]/strat_train_set[locc].value_counts()[1])
print('in testing set  :',strat_test_set[locc].value_counts()[0]/strat_test_set[locc].value_counts()[1])

#%%
## -- generate training data and labels of training data
data_ = strat_train_set.drop('subject', axis=1)
data_labels = strat_train_set['subject'].copy()
datatest_ = strat_test_set.drop('subject', axis=1)
datatest_labels = strat_test_set['subject'].copy()

#%% -------------------------------------------------------------------
## ----- train model with data
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


#%% -------------------------------------------------------------------
### -----RandomForestClassifier
# print(' \n ---------------------  ')
# log_clf = LogisticRegression()
# rnd_clf = RandomForestClassifier()
rnd_clf = RandomForestClassifier(random_state=random_state)
rnd_clf.fit(data_, data_labels)
X_test = datatest_
y_test = datatest_labels
predictions_ = rnd_clf.predict(X_test)
acc = accuracy_score(y_test, predictions_)
print('RandomForestClassifier accuracy: ',acc)

# #%%  --- save the model ----------------
# print('save the model')
# modelfile = "model_questionaire.pickle"
# pickle.dump(rnd_clf, open(modelfile, "wb"))

# #%%  --- load the model and test
# print('test test set')
# loaded_model = pickle.load(open(modelfile, "rb"))
# predictions_ = loaded_model.predict(X_test)
# acc = accuracy_score(y_test, predictions_)
# print('test load model and predict: accuracy: ',acc)

# print('test sub set')
# X_test_sub =  X_test.iloc[[20]]
# y_test_sub = y_test.iloc[[20]]
# predictions_ = loaded_model.predict(X_test_sub)
# acc = accuracy_score(y_test_sub, predictions_)
# print('test load model and predict: accuracy: ',acc)


# --- save model ---
import joblib
# modelfile = "model_questionaire2.pkl"
# joblib.dump(rnd_clf,  modelfile)

# --- load model ---
loaded_model = joblib.load(modelfile) 

predictions_ = loaded_model.predict(X_test)
acc = accuracy_score(y_test, predictions_)
print('test load model and predict: accuracy: ',acc)

print('test sub set')
X_test_sub =  X_test.iloc[[20]]
y_test_sub = y_test.iloc[[20]]
predictions_ = loaded_model.predict(X_test_sub)
acc = accuracy_score(y_test_sub, predictions_)
print('test load model and predict: accuracy: ',acc)