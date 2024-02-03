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
import math 

ENCODING = 'utf-8'
# loaded_model = joblib.load('./model_only_va.joblib')
import pickle
# save the iris classification model as a pickle file
# --- load model ---
model_pkl_file = "Model_pickle_questionaire_1_only_va.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_q = pickle.load(file) 
# --- load model ---
model_pkl_file = "Model_pickle_dualtap_1_only_va.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_d = pickle.load(file) 


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
#         read_feature = req['feature'] #if read a list
#         df3 = pd.DataFrame([read_feature])
#         df4 = df3.values[0]
#         return("read_feature type: "+type(read_feature)+":"+len(read_feature)+" "+type(df3)+":"+len(df3)+" "+type(df4))+":"+len(df4)


@app.route('/readjson_patientName', methods=['POST'])  
def readjsonpatientName():
    if request.is_json:
        req = request.get_json()
        read_patientName = req['patientName'] #if read a list
        return("patientName : "+read_patientName)

@app.route('/readjson_feat', methods=['POST'])  
def readjson_feat():
    if request.is_json:
        req = request.get_json()
        read_feat = req['feature'] #if read a list
        return("read_feat : "+read_feat)

@app.route('/predict_questionaire', methods=['POST'])  
def predict_questionaire():
    if request.is_json:
        req = request.get_json()
        #read the request as web read in --------------------------------
        read_feat = req['feature'] #readin as string, need convert to list of float
        # handle to list of float
        list_of_integers = [
            float(item) if item.isdigit() else item
            for item in read_feat.split(',')
        ]
        df3 = pd.DataFrame([list_of_integers])
        predictions_ = loaded_model_q.predict(df3.values)
        # return ("predictin: "+predictions_[0])
        return jsonify({"prediction":str(predictions_[0])}) 
        # return ("predictin: "+str(predictions_[0]))




@app.route('/predict_dualtap_fromFeature', methods=['POST'])  
def predict_dualtap_fromFeature():
    if request.is_json:
        req = request.get_json()
        #read the request as web read in --------------------------------
        read_feat = req['feature'] #readin as string, need convert to list of float
        # handle to list of float
        list_of_integers = [
            float(item) if item.isdigit() else item
            for item in read_feat.split(',')
        ]
        df3 = pd.DataFrame([list_of_integers])
        predictions_ = loaded_model_d.predict(df3.values)
        # return ("predictin: "+predictions_[0])
        return jsonify({"prediction":str(predictions_[0])}) 
        # return ("predictin: "+str(predictions_[0]))
    



@app.route('/predict_dualtap', methods=['POST'])  
def predict_dualtap():
    if request.is_json:
        req = request.get_json()
        #read the request as web read in --------------------------------
        read_feat = req['feature'] #readin as string, need convert to list of float
        # handle to list of float
        list_of_integers = [
            float(item) if item.isdigit() else item
            for item in read_feat.split(',')
        ]
        df3 = pd.DataFrame([list_of_integers])
        predictions_ = loaded_model_d.predict(df3.values)
        # return ("predictin: "+predictions_[0])
        return jsonify({"prediction":str(predictions_[0])}) 
        # return ("predictin: "+str(predictions_[0]))
    

@app.route('/predict_dualtap_featureprepare', methods=['POST'])  
def predict_dualtap_featureprepare():
    if request.is_json:
        data = request.get_json()
        i = data['score']
        aScor = i

        TapCount = []
        j = data['recording']['taps']
        for jj in j:
            da = jj['data']
            # print(len(da))
            TapCount.append(len(da))

        maxTapCount  = np.max(TapCount)
        meanTapCount = np.mean(TapCount)
        stdTapCount  = np.std(TapCount)

        row = []
        

        # --------------  about circle ---------------
        jL = data['recording']['circleL']
        jR = data['recording']['circleR']
        cLx,cLy = jL['x'],jL['y']
        cRx,cRy = jR['x'],jR['y']
        cRadius = jL['r']

        # --- check touched points in or out as proportional 
        distancEachPoint = []
        inLeft = 0
        inRight = 0
        jf = data['recording']['taps']
        ts_instroke_mean = []
        for jstroke in jf:
            jjj = jstroke['data']
            ts_instroke = [] 
            for jjjj in jjj:
                xpo,ypo,ts_ = jjjj['x'],jjjj['y'],jjjj['ts']
                ts_instroke.append(ts_)
                # compare to L and R, then decide which side
                if (xpo-cLx) >= (cRx-xpo):
                    # point is in the right:
                    inRight +=1
                    dispo = math.sqrt((xpo-cRx)*(xpo-cRx) + (ypo-cRy)*(ypo-cRy))
                    distancEachPoint.append(dispo)
                else:
                    # point is in the left:
                    inLeft += 1
                    dispo = math.sqrt((xpo-cLx)*(xpo-cLx) + (ypo-cLy)*(ypo-cLy))
                    distancEachPoint.append(dispo)

                # print(da['x'],da['y'],dispo)
            ts_instroke_mean.append(mean(ts_instroke))
            # ts_instroke_mean = mean(ts_instroke)
        countinside = sum(map(lambda x : x < cRadius, distancEachPoint))
        countall = len(distancEachPoint)
        # print(countinside, countall)
        ppInsideToAll = countinside/countall
        try:
            if inLeft/inRight > 1:
                ppLeftToRight =inRight/inLeft
            else:
                ppLeftToRight = inLeft/inRight
        except:
            ppLeftToRight = 0

        # about time diff
        tDiff = list()
        for item1, item2 in zip(ts_instroke_mean[1:], ts_instroke_mean[0:-1]):
            item = item1 - item2
            tDiff.append(item)        
        tDiff_mean = mean(tDiff)
        tDiff_max = max(tDiff)
        tDiff_min = min(tDiff)

        E0 = '%.5f'%(aScor)   
        E1 = '%.5f'%(maxTapCount)           
        E2 = '%.5f'%(meanTapCount)                     
        E3 = '%.5f'%(stdTapCount)
        E4 = '%.5f'%(countall)
        E5 = '%.5f'%(ppInsideToAll)
        E6 = '%.5f'%(ppLeftToRight)   
        E7 = '%.5f'%(tDiff_mean)
        E8 = '%.5f'%(tDiff_max)
        E9 = '%.5f'%(tDiff_min)
        rowx = [E0,E1,E2,E3,E4,E5,E6,E7,E8,E9]

        return jsonify({"aScor":E0},{"sometig2":E1}) 
        # return ("predictin: "+str(predictions_[0]))
    



@app.route('/readjson_feat2', methods=['POST'])  
def readjson_feat2():
    # if request.is_json:
    #     req = request.get_json()
    #     #read the request as web read in --------------------------------
    #     read_feat = req['feature'] #readin as string, need convert to list of float
    #     # # handle to list of float
    #     # list_of_integers = [
    #     #     float(item) if item.isdigit() else item
    #     #     for item in read_feat.split(',')
    #     # ]
    #     # df3 = pd.DataFrame([list_of_integers])
    #     # df3 = pd.DataFrame([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    #     # predictions_ = loaded_model.predict(df3.values)
    #     return (0)
    if request.is_json:
        req = request.get_json()
        read_feat = req['feature'] #if read a list
        return(read_feat)
    


@app.route('/readjson_feat_do2', methods=['POST'])  
def readjson_feat_do2():
    if request.is_json:
        req = request.get_json()
        read_feat = req['feature'] #if read a list
        df3 = pd.DataFrame([read_feat])
        jsondf3 = df3.to_json() 
        return(jsondf3)




@app.route('/readjson', methods=['POST'])  
def readjson():
    try:
        if request.is_json:
            req = request.get_json()
            read_feature = req['feature'] #if read a list
            df3 = pd.DataFrame([read_feature])
            df4 = df3.values[0]
            return("read_feature type: "+
                   type(read_feature)+":"+len(read_feature)+" "+
                   type(df3)+":"+len(df3)+" "+
                   type(df4)+":"+len(df4)
                   )
    except BaseException:
        return(logging.exception("An exception was thrown!"))





@app.route('/simpleML', methods=['POST'])  
def simpleML():
    if request.is_json:
        req = request.get_json() 
        # feature = req['feature']
        # return("feature "+feature[0]+" "+feature[19])
        # req = json.loads(z)

        feature = req['feature']
        # df = pd.DataFrame([feature])
        # predictions_ = loaded_model.predict(df.values)
        # returnPredict = predictions_[0]
        return("prediction output: xxxxxxxxxxxxxxxxx" )

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

