# lib need to install, in requirement.txt
from flask import Flask, jsonify
from flask import render_template
from flask import request, Response, send_file, redirect, safe_join, abort
import pandas as pd
import numpy as np
from numpy import mean, sqrt, square, arange
from numpy import genfromtxt
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.signal import welch, hann
import requests

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import pickle


# built-in lib
import time, threading, csv, random, json
from datetime import datetime, timedelta
import unicodedata
import re
import os
import shutil
import time
import random
import datetime
import base64
from decimal import Decimal
import gzip
import io
from io import BytesIO
import zipfile
from zipfile import ZipFile 
import urllib.parse
import logging
import traceback
import joblib

ENCODING = 'utf-8'
# loaded_model = joblib.load('./model_only_va.joblib')
import pickle
# save the iris classification model as a pickle file
model_pkl_file = "Model_pickle_1.pkl"  
# with open(model_pkl_file, 'wb') as file:  
#     pickle.dump(rnd_clf, file)
# --- load model ---
with open(model_pkl_file, 'rb') as file:  
    loaded_model = pickle.load(file)



app = Flask(__name__)

@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})


@app.route('/')
def saysomething():
    return ("now what ------------777788")


# @app.route('/readjson', methods=['POST'])  
# def readjson():
#     # --- load model ---
#     # modelfile = "model_questionaire2.joblib"
#     # loaded_model = joblib.load("model_questionaire2.joblib")
#     if request.is_json:
#         req = request.get_json()
#         read_pName = req['patientName']
#         return("patient name: "+read_pName)


# @app.route('/simpleML', methods=['POST'])  
# def simpleML():
#     if request.is_json:
#         req = request.get_json()
#         # feature = req['feature']
#         # return("feature "+feature[0]+" "+feature[19])
#         # req = json.loads(z)

#         feature = req['feature']
#         # df = pd.DataFrame([feature])
#         # predictions_ = loaded_model.predict(df.values)
#         # returnPredict = predictions_[0]
#         return("prediction output: " )

# @app.route('/getcwd', methods=['POST'])  
# def getcwd():
#     # --- load model ---
#     return(os.getcwd())


# predictions_ = loaded_model.predict(X_test)
# acc = accuracy_score(y_test, predictions_)
# print('test load model and predict: accuracy: ',acc)

# print('test sub set')
# X_test_sub =  X_test.iloc[[20]]
# y_test_sub = y_test.iloc[[20]]
# predictions_ = loaded_model.predict(X_test_sub)
# acc = accuracy_score(y_test_sub, predictions_)
# print('test load model and predict: accuracy: ',acc)

