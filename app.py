# lib need to install, in requirement.txt
from flask import Flask, jsonify
from flask import render_template
from flask import request, Response, send_file, redirect, safe_join, abort
import pandas as pd
import numpy as np
from numpy import mean, sqrt, square, arange
from numpy import genfromtxt
from scipy import signal
from scipy.stats import entropy
from scipy.stats import kurtosis, skew
from scipy.fft import fft, rfft
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.signal import welch, hann
from scipy.stats.mstats import zscore

import requests
import parselmouth 
import statistics
from moviepy.editor import *

from parselmouth.praat import call


from sklearn.preprocessing import StandardScaler
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
import EntropyHub as EH

from cryptography.fernet import Fernet

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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# design hipass filter
sos = signal.butter(0, 1, 'hp', fs=10, output='sos')

# load key
f = open("readke", mode="rb")
masterkey = f.read()
f.close() 
fernet = Fernet(masterkey)

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
# --- load model ---
model_pkl_file = "Model_pickle_pinchtosize_1_only_va.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_p = pickle.load(file) 
# --- load model ---
model_pkl_file = "Model_tremor_rest_a_only_001.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_tr = pickle.load(file) 
# --- load model ---
model_pkl_file = "Model_tremor_post_a_only_001.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_tp = pickle.load(file) 
# --- load model ---
model_pkl_file = "Model_pickle_gait_stab_1.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_gs = pickle.load(file) 
# --- load model ---
model_pkl_file = "Model_pickle_gait_walk_1.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_gw = pickle.load(file) 
# --- load model ---
# model_pkl_file = "Model_pickle_voic_ahh_15.pkl"  
# with open(model_pkl_file, 'rb') as file:  
#     loaded_model_va = pickle.load(file) 
# --- load model ---
model_pkl_file = "Model_pickle_voice_ypl_1.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_vy = pickle.load(file) 


# --- load model --- > modified/improved model ahh voice
model_pkl_file = "Model_pickle_voic_ahh_imp_001.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_va_imp = pickle.load(file) 
# --- load model --- > modified/improved model ypl voice
model_pkl_file = "Model_pickle_voic_ypl_imp_001.pkl"  
with open(model_pkl_file, 'rb') as file:  
    loaded_model_vy_imp = pickle.load(file) 

app = Flask(__name__)

@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})


@app.route('/')
def saysomething():
    return ("now what ------------777788")



def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    # meanF0 = call(pitch, "Get maximum", 0, 700, "Hertz", "Parabolic") # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

# This function measures formants using Formant Position formula
def measureFormants(sound, wave_file, f0min,f0max):
    sound = parselmouth.Sound(sound) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)
    
    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)
    
    return f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median

def runPCA(df):
    # z-score the Jitter and Shimmer measurements
    measures = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    x = df.loc[:, measures].values
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
    principalDf
    return principalDf


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
    try:
        if request.is_json:
            req = request.get_json()
            #read the request as web read in --------------------------------
            read_feat = req['data'] #readin as string, need convert to list of float
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
    except:
        return jsonify({"prediction":str(2)})     



@app.route('/predict_dualtap', methods=['POST'])  
def predict_dualtap():
    try:
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

            # row = []
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
            df3 = pd.DataFrame([rowx])
            predictions_ = loaded_model_d.predict(df3.values)
            # return ("predictin: "+predictions_[0])
            return jsonify({"prediction":str(predictions_[0])}) 
    except:
        return jsonify({"prediction":str(2)})     




@app.route('/predict_pinchtosize', methods=['POST'])  
def predict_pinchtosize():
    try:
        if request.is_json:
            data = request.get_json()
            CountAllHandOff = 0
            allSignSwitchCount = 0
            mdiaOuter = []

            x_mts_start_stack = []
            x_mts_end_stack = []

            x_norm_std_stack = []
            x_norm_mean_stack = []

            x_mx1_std_stack = []
            x_mx1_mean_stack = []

            x_mx2_std_stack = []
            x_mx2_mean_stack = []

            x_my1_std_stack = []
            x_my1_mean_stack = []
            
            x_my2_std_stack = []
            x_my2_mean_stack = []
            

            i = data['data']
            for indata in i:
                j_meta = indata['meta']
                print('----- each meta------')
                # print('fr : ',fr)
                # print('target : ',j_meta['target'])
                j_indata = indata['data']
                # fig, ax = plt.subplots()
                # mts,mx1,mx2,my1,my2,mdia,mcx,mcy=[],[],[],[],[],[],[],[]
                for k_injdata in j_indata:
                    CountAllHandOff += 1
                    # print('count  : ',len(k_injdata))
                    mts,mx1,mx2,my1,my2,mdia,mcx,mcy=[],[],[],[],[],[],[],[]
                    print('   --> each data  ')
                    for m_inkdata in k_injdata:
                        mts.append(m_inkdata['timestamp'])
                        mx1.append(m_inkdata['x1'])
                        mx2.append(m_inkdata['x2'])
                        my1.append(m_inkdata['y1'])
                        my2.append(m_inkdata['y2'])
                        mdia.append(m_inkdata['diameter'])
                        mcx.append(m_inkdata['center']['x'])
                        mcy.append(m_inkdata['center']['y'])
                    # -- try some smooth
                    mvWindow = 25
                    if len(mdia) > mvWindow+1:
                        # fig, ax = plt.subplots()
                        # fig.set_size_inches(3, 3)
                        mav = moving_average(mdia, 25)
                        # -- count sign change of comparison
                        signSwitchCount = 0
                        for idx,dat_ in enumerate(mav):
                            if idx == 0:
                                mvToRaw_a = (dat_>mdia[idx])
                            else:
                                mvToRaw_b = (dat_>mdia[idx])
                                if mvToRaw_b != mvToRaw_a:
                                    signSwitchCount +=1
                                    mvToRaw_a = mvToRaw_b
                        allSignSwitchCount = allSignSwitchCount + signSwitchCount

                mts_ar = np.array(mts)
                mts_start = np.array(mts[0])
                mts_end = np.array(mts[-1])
                print(mts_start, mts_end)

                mdia_ar = np.array(mdia)
                x_norm_ar = (mdia_ar-np.min(mdia_ar))/(np.max(mdia_ar)-np.min(mdia_ar))
                x_norm_std = np.std(x_norm_ar)
                x_norm_mean = np.mean(x_norm_ar)
                # x_norm_std = np.std(mdia_ar)
                # x_norm_mean = np.mean(mdia_ar)

                mdia_ar = np.array(mx1)
                x_norm_ar = (mdia_ar-np.min(mdia_ar))/(np.max(mdia_ar)-np.min(mdia_ar))
                x_mx1_std = np.std(x_norm_ar)
                x_mx1_mean = np.mean(x_norm_ar)
                # x_mx1_std = np.std(mdia_ar)
                # x_mx1_mean = np.mean(mdia_ar)                

                mdia_ar = np.array(mx2)
                x_norm_ar = (mdia_ar-np.min(mdia_ar))/(np.max(mdia_ar)-np.min(mdia_ar))
                x_mx2_std = np.std(x_norm_ar)
                x_mx2_mean = np.mean(x_norm_ar)
                # x_mx2_std = np.std(mdia_ar)
                # x_mx2_mean = np.mean(mdia_ar)

                mdia_ar = np.array(my1)
                x_norm_ar = (mdia_ar-np.min(mdia_ar))/(np.max(mdia_ar)-np.min(mdia_ar))
                x_my1_std = np.std(x_norm_ar)
                x_my1_mean = np.mean(x_norm_ar)
                # x_my1_std = np.std(mdia_ar)
                # x_my1_mean = np.mean(mdia_ar)

                mdia_ar = np.array(my2)
                x_norm_ar = (mdia_ar-np.min(mdia_ar))/(np.max(mdia_ar)-np.min(mdia_ar))
                x_my2_std = np.std(x_norm_ar)
                x_my2_mean = np.mean(x_norm_ar)
                # x_my2_std = np.std(mdia_ar)
                # x_my2_mean = np.mean(mdia_ar)

                # ----------- append to create over all  -----------
                x_mts_start_stack.append(mts_start)
                x_mts_end_stack.append(mts_end)

                x_norm_std_stack.append(x_norm_std)
                x_norm_mean_stack.append(x_norm_mean)

                x_mx1_std_stack.append(x_mx1_std)
                x_mx1_mean_stack.append(x_mx1_mean)

                x_mx2_std_stack.append(x_mx2_std)
                x_mx2_mean_stack.append(x_mx2_mean)

                x_my1_std_stack.append(x_my1_std)
                x_my1_mean_stack.append(x_my1_mean)
                
                x_my2_std_stack.append(x_my2_std)
                x_my2_mean_stack.append(x_my2_mean)
                                                                

            # in each file: record ...
            mts_start_mean = np.mean(x_mts_start_stack)
            mts_start_max = np.max(x_mts_start_stack)
            mts_range_max = np.max(x_mts_end_stack)

            mdia_std = np.mean(x_norm_std_stack)
            # mdia_mean = np.mean(x_norm_mean_stack)

            mx1_std = np.mean(x_mx1_std_stack)
            # mx1_mean = np.mean(x_mx1_mean_stack)

            mx2_std = np.mean(x_mx2_std_stack)
            # mx2_mean = np.mean(x_mx2_mean_stack)

            my1_std = np.mean(x_my1_std_stack)
            # my1_mean = np.mean(x_my1_mean_stack)

            my2_std = np.mean(x_my2_std_stack)
            # my2_mean = np.mean(x_my2_mean_stack)


            # mdiaOuter.append(x_norm_std)
            row = []

            E0 = '%.5f'%(mdia_std)
            # E1 = '%.5f'%(mdia_mean)
            E2 = '%.5f'%(mx1_std)
            # E3 = '%.5f'%(mx1_mean)
            E4 = '%.5f'%(mx2_std)
            # E5 = '%.5f'%(mx2_mean)
            E6 = '%.5f'%(my1_std)
            # E7 = '%.5f'%(my1_mean)
            E8 = '%.5f'%(my2_std)
            # E9 = '%.5f'%(my2_mean)
            # E10 = '%.5f'%(CountAllHandOff)
            E11 = '%.5f'%(allSignSwitchCount)
            E12 = '%.5f'%(mts_start_mean)
            E13 = '%.5f'%(mts_range_max)
            E14 = '%.5f'%(mts_start_max)

            rowx = [E0,E2,E4,E6,E8,E11,E12,E13,E14]
            df3 = pd.DataFrame([rowx])
            predictions_ = loaded_model_p.predict(df3.values)
            # return jsonify({"mdia_std":E0,
            #                 "mx1_std":E2,
            #                 "mx2_std":E4,
            #                 "my1_std":E6,
            #                 "my2_std":E8,
            #                 "allSignSwitchCount":E11,
            #                 "mts_start_mean":E12,
            #                 "mts_range_max":E13,
            #                 "mts_start_max":E14                      
            #                 }) 
            return jsonify({"prediction":str(predictions_[0])}) 
    except:
        return jsonify({"prediction":str(2)})     




@app.route('/predict_gait_stab', methods=['POST'])  
def predict_gait_stab():
    try:
        if request.is_json:
            data = request.get_json()
            tStamp = []
            acX = []
            acY = []
            acZ = []
            agX = []
            agY = []
            agZ = []

            for i in data['recording']['recordedData']:
                tsC = i['ts']
                tStamp.append(tsC)
                acXC = i['data'][0]
                acYC = i['data'][1]
                acZC = i['data'][2]    
                acX.append(acXC)
                acY.append(acYC)
                acZ.append(acZC) 

                agXC = i['data'][3]
                agYC = i['data'][4]
                agZC = i['data'][5]    
                agX.append(agXC)
                agY.append(agYC)
                agZ.append(agZC) 

            tst = [item - tStamp[0] for item in tStamp]
            # ------------  handle the oversampling to 200 samples in 20 sec
            if len(acX) > 200:
                toBeSamp = 200
                # print('----> ' + str(filepath))
                acX, x1 = signal.resample(acX,toBeSamp,np.arange(len(acX)))  # resampled at 200
                acY, x1 = signal.resample(acY,toBeSamp,np.arange(len(acY)))  # resampled 
                acZ, x1 = signal.resample(acZ,toBeSamp,np.arange(len(acZ)))  # resampled 
                agX, x1 = signal.resample(agX,toBeSamp,np.arange(len(agX)))  # resampled 
                agY, x1 = signal.resample(agY,toBeSamp,np.arange(len(agY)))  # resampled
                agZ, x1 = signal.resample(agZ,toBeSamp,np.arange(len(agZ)))  # resampled


            row = []
            for testsig in (acX,acY,acZ,agX,agY,agZ):
            # for testsig in (acX,acY,acZ):
                testsig_filt = signal.sosfilt(sos, testsig)
                res = np.array(testsig_filt)
                fourier = fft(testsig_filt)
                fab = np.abs(fourier)[0:100]
                # ------------ 
                Esum = sum(np.square(fab))
                F1 = sum(np.square(fab[0:25]))
                F2 = sum(np.square(fab[25:50]))
                F3 = sum(np.square(fab[50:75]))
                F4 = sum(np.square(fab[75:80]))
                kur = kurtosis(testsig_filt, fisher=True)
                ske = skew(testsig_filt, bias=False)
                resdif = res[1:]-res[0:-1]
                Mobi = np.sqrt(np.var(resdif)/np.var(res))
                resdif2 = resdif[1:]-resdif[0:-1]
                compx = np.sqrt(np.var(resdif2)*np.var(res)/(np.var(resdif)*np.var(resdif)))
                
                # E1 = '%.5f'%(F1/Esum)
                E1 = '%.5f'%(np.std(testsig_filt))           
                E2 = '%.5f'%(np.mean(testsig_filt))                     
                E3 = '%.5f'%(kur)
                E4 = '%.5f'%(ske)
                E5 = '%.5f'%(Mobi)   
                E6 = '%.5f'%(compx)                              
                E7 = '%.5f'%(F1) 
                E8 = '%.5f'%(F2) 
                E9 = '%.5f'%(F3) 
                E10 = '%.5f'%(F4) 
                E11 = '%.5f'%(F2/Esum)
                E12 = '%.5f'%(F3/Esum)

                Samp, Phi1, Phi2 = EH.SampEn(res, m = 2, tau = 2)
                E13 = '%.5f'%(Samp[0])
                E14 = '%.5f'%(Samp[1])
                E15 = '%.5f'%(Samp[2])
                rowx = [E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15]
                row = row + rowx
            
            toListofNumber = [float(x) for x in row]
            X = np.array([toListofNumber])
            predictions_ = loaded_model_gs.predict(X)        
            return jsonify({"prediction":str(predictions_[0])}) 
    except:
        return jsonify({"prediction":str(2)}) 


@app.route('/predict_gait_walk', methods=['POST'])  
def predict_gait_walk():
    try:
        if request.is_json:
            data = request.get_json()
            tStamp = []
            acX = []
            acY = []
            acZ = []
            agX = []
            agY = []
            agZ = []

            for i in data['motion']:
                # tsC = i['ts']
                # tStamp.append(tsC)
                acXC = i['data'][0]
                acYC = i['data'][1]
                acZC = i['data'][2]    
                acX.append(acXC)
                acY.append(acYC)
                acZ.append(acZC) 

                agXC = i['data'][3]
                agYC = i['data'][4]
                agZC = i['data'][5]    
                agX.append(agXC)
                agY.append(agYC)
                agZ.append(agZC) 

            tst = [item - tStamp[0] for item in tStamp]

            # ------------  handle the oversampling to 200 samples in 20 sec
            if len(acX) > 200:
                toBeSamp = 200
                # print('----> ' + str(filepath))
                acX, x1 = signal.resample(acX,toBeSamp,np.arange(len(acX)))  # resampled at 200
                acY, x1 = signal.resample(acY,toBeSamp,np.arange(len(acY)))  # resampled 
                acZ, x1 = signal.resample(acZ,toBeSamp,np.arange(len(acZ)))  # resampled 
                agX, x1 = signal.resample(agX,toBeSamp,np.arange(len(agX)))  # resampled 
                agY, x1 = signal.resample(agY,toBeSamp,np.arange(len(agY)))  # resampled
                agZ, x1 = signal.resample(agZ,toBeSamp,np.arange(len(agZ)))  # resampled

            row = []
            for testsig in (acX,acY,acZ,agX,agY,agZ):
            # for testsig in (acX,acY,acZ):
                testsig_filt = signal.sosfilt(sos, testsig)
                res = np.array(testsig_filt)
                fourier = fft(testsig_filt)
                fab = np.abs(fourier)[0:100]
                # ------------ 
                Esum = sum(np.square(fab))
                # Esum = 1.0
                # base = 2  # work in units of bits
                F1 = sum(np.square(fab[0:25]))
                F2 = sum(np.square(fab[25:50]))
                F3 = sum(np.square(fab[50:75]))
                F4 = sum(np.square(fab[75:80]))

                kur = kurtosis(testsig_filt, fisher=True)
                ske = skew(testsig_filt, bias=False)
                resdif = res[1:]-res[0:-1]
                Mobi = np.sqrt(np.var(resdif)/np.var(res))
                resdif2 = resdif[1:]-resdif[0:-1]
                compx = np.sqrt(np.var(resdif2)*np.var(res)/(np.var(resdif)*np.var(resdif)))

                E1 = '%.5f'%(np.std(testsig_filt))           
                E2 = '%.5f'%(np.mean(testsig_filt))                     
                E3 = '%.5f'%(kur)
                E4 = '%.5f'%(ske)
                E5 = '%.5f'%(Mobi)   
                E6 = '%.5f'%(compx)                              
                E7 = '%.5f'%(F1) 
                E8 = '%.5f'%(F2) 
                E9 = '%.5f'%(F3) 
                E10 = '%.5f'%(F4) 
                E11 = '%.5f'%(F2/Esum)
                E12 = '%.5f'%(F3/Esum)
                E13 = '%.5f'%(np.var(resdif2))
                rowx = [E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13]
                row = row + rowx
            
            toListofNumber = [float(x) for x in row]
            X = np.array([toListofNumber])
            predictions_ = loaded_model_gw.predict(X)        
            return jsonify({"prediction":str(predictions_[0])}) 
    except:
        return jsonify({"prediction":str(2)}) 

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


# +++++++++++++++++++++++++++++++++
#
# +++++++++++++++++++++++++++++++++
@app.route('/predict_voting', methods=['POST'])  
def predict_voting():
    filenamecombination = 'listof_best_combination.csv'
    # create lists to put the results
    try:
        if request.is_json:
            jsonData = request.get_json()
            test_available = jsonData["test_available"]     
            test_result = jsonData["test_result"] 
            test_result_number = [
                float(item) if item.isdigit() else item
                for item in test_result.split(',')
            ]

            # test_available = ['ques', 'voiS', 'voiA', 'post', '', '', 'stbl', '', '']
            # test_result = ['0', '1', '1', '1', '-1', '-1', '0', '-1', '-1']
            with open(filenamecombination) as file_obj: 
                reader_obj = csv.reader(file_obj) 
                # Iterate over each row in the csv  
                for row in reader_obj: 
                    phrase_to_list = row[0].split("\t")
                    if (phrase_to_list==test_available):
                        # return jsonify({"prediction":str(1)}) 
                        # test_result_number = [ int(x) for x in test_result ]
                        if test_result_number.count(1) >= test_result_number.count(0):
                            return jsonify({"prediction":str(1)}) 
                        else:
                            return jsonify({"prediction":str(0)}) 
    except:
        return jsonify({"prediction":str(2)}) 



# +++++++++++++++++++++++++++++++++
#
# +++++++++++++++++++++++++++++++++
@app.route('/predict_voice_ahh', methods=['POST'])  
def predict_voice_ahh():
    # create lists to put the results
    file_list = []
    duration_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []
    f1_mean_list = []
    f2_mean_list = []
    f3_mean_list = []
    f4_mean_list = []
    f1_median_list = []
    f2_median_list = []
    f3_median_list = []
    f4_median_list = []

    try:
        if request.is_json:
            jsonData = request.get_json()
            pName = jsonData["patientName"]
            voicebase64 = jsonData["data"]
            wavFilename = pName+"_temp.wav"
            wavFile= open(wavFilename, "wb")
            voicedecoded = base64.b64decode(voicebase64)
            wavFile.write(voicedecoded)
            wave_file = wavFilename

            sound = parselmouth.Sound(wave_file)
            (duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, 
            localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(
                sound, 75, 300, "Hertz")
            (f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median) = measureFormants(
                sound, wave_file, 75, 300)
            # file_list.append(filename_) # make an ID list
            # duration_list.append(duration) # make duration list
            mean_F0_list.append(meanF0) # make a mean F0 list
            sd_F0_list.append(stdevF0) # make a sd F0 list
            hnr_list.append(hnr) #add HNR data
            
            # add raw jitter and shimmer measures
            localJitter_list.append(localJitter)
            localabsoluteJitter_list.append(localabsoluteJitter)
            rapJitter_list.append(rapJitter)

            ppq5Jitter_list.append(ppq5Jitter)
            ddpJitter_list.append(ddpJitter)
            localShimmer_list.append(localShimmer)

            localdbShimmer_list.append(localdbShimmer)
            apq3Shimmer_list.append(apq3Shimmer)
            aqpq5Shimmer_list.append(aqpq5Shimmer)

            apq11Shimmer_list.append(apq11Shimmer)
            ddaShimmer_list.append(ddaShimmer)
            
            # add the formant data
            f1_mean_list.append(f1_mean)
            f2_mean_list.append(f2_mean)
            f3_mean_list.append(f3_mean)
            f4_mean_list.append(f4_mean)

            f1_median_list.append(f1_median)
            f2_median_list.append(f2_median)
            f3_median_list.append(f3_median)
            f4_median_list.append(f4_median)


            # //  +++++++++++++++++++++++++++++
            fdisp = []
            for i in range(len(f4_median_list)):
                fdisp.append((f4_median_list[i] - f1_median_list[i])/3)

            avgFormant = []
            for i in range(len(f4_median_list)):
                avgFormant.append((f1_median_list[i] + f2_median_list[i] + f3_median_list[i] + f4_median_list[i]) /4)

            mff = []
            for i in range(len(f4_median_list)):
                mff.append((f1_median_list[i] * f2_median_list[i] * f3_median_list[i] * f4_median_list[i]) **0.25)

            fitch_vtl = []
            for i in range(len(f4_median_list)):
                fitch_vtl.append(((1 * (35000 / (4 * f1_median_list[i]))) +
                                (3 * (35000 / (4 * f2_median_list[i]))) + 
                                (5 * (35000 / (4 * f3_median_list[i]))) + 
                                (7 * (35000 / (4 * f4_median_list[i])))) / 4 )

            xysum = []
            for i in range(len(f4_median_list)):
                xysum.append((0.5 * f1_median_list[i]) + 
                            (1.5 * f2_median_list[i]) + 
                            (2.5 * f3_median_list[i]) + 
                            (3.5 * f4_median_list[i]))

            xsquaredsum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
            delta_f = []
            for i in range(len(f4_median_list)):
                delta_f.append(xysum[i]/xsquaredsum)

            vtl_delta_f = []
            for i in range(len(f4_median_list)):
                vtl_delta_f.append(35000 / (2 * delta_f[i]))

            # subject = []
            # for i in range(len(f4_median_list)):
            #     subject.append(mark_)
            # //  +++++++++++++++++++++++++++++


            # Add the data to Pandas
            df = pd.DataFrame(np.column_stack([  mean_F0_list, sd_F0_list, hnr_list, 
                                            localJitter_list, localabsoluteJitter_list, rapJitter_list, 
                                            ppq5Jitter_list, ddpJitter_list, localShimmer_list, 
                                            localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, 
                                            apq11Shimmer_list, ddaShimmer_list, 
                                            f1_mean_list, f2_mean_list, f3_mean_list, f4_mean_list, 
                                            f1_median_list, f2_median_list, f3_median_list, f4_median_list,
                                            fdisp, avgFormant,mff,fitch_vtl,xysum,delta_f,vtl_delta_f
                                            ]),
                                            columns=[ 'meanF0Hz', 'stdevF0Hz', 'HNR', 
                                                        'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                                        'ppq5Jitter', 'ddpJitter', 'localShimmer', 
                                                        'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                                        'apq11Shimmer', 'ddaShimmer', 
                                                        'f1_mean', 'f2_mean', 'f3_mean', 'f4_mean', 
                                                        'f1_median', 'f2_median', 'f3_median', 'f4_median', 
                                                        'fdisp','avgFormant','mff','fitch_vtl','xysum','delta_f','vtl_delta_f'                                                    
                                                        ])

            X = np.array(df)
            predictions_ = loaded_model_va_imp.predict(X)        
            # return jsonify({"show something":str('this is something')}) 
            return jsonify({"prediction":str(predictions_[0])}) 
    except:
        return jsonify({"prediction":str(2)})     


# +++++++++++++++++++++++++++++++++
#
# +++++++++++++++++++++++++++++++++
@app.route('/predict_voice_ypl', methods=['POST'])  
def predict_voice_ypl():
    # create lists to put the results
    file_list = []
    duration_list = []
    mean_F0_list = []
    sd_F0_list = []
    hnr_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    ddpJitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    ddaShimmer_list = []
    f1_mean_list = []
    f2_mean_list = []
    f3_mean_list = []
    f4_mean_list = []
    f1_median_list = []
    f2_median_list = []
    f3_median_list = []
    f4_median_list = []

    try:
        if request.is_json:
            jsonData = request.get_json()
            pName = jsonData["patientName"]
            voicebase64 = jsonData["data"]
            wavFilename = pName+"_temp.wav"
            wavFile= open(wavFilename, "wb")
            voicedecoded = base64.b64decode(voicebase64)
            wavFile.write(voicedecoded)
            wave_file = wavFilename

            sound = parselmouth.Sound(wave_file)
            (duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, 
            localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(
                sound, 75, 300, "Hertz")
            (f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median) = measureFormants(
                sound, wave_file, 75, 300)
            
            os.remove(wave_file)

            # file_list.append(filename_) # make an ID list
            # duration_list.append(duration) # make duration list
            mean_F0_list.append(meanF0) # make a mean F0 list
            sd_F0_list.append(stdevF0) # make a sd F0 list
            hnr_list.append(hnr) #add HNR data
            
            # add raw jitter and shimmer measures
            localJitter_list.append(localJitter)
            localabsoluteJitter_list.append(localabsoluteJitter)
            rapJitter_list.append(rapJitter)

            ppq5Jitter_list.append(ppq5Jitter)
            ddpJitter_list.append(ddpJitter)
            localShimmer_list.append(localShimmer)

            localdbShimmer_list.append(localdbShimmer)
            apq3Shimmer_list.append(apq3Shimmer)
            aqpq5Shimmer_list.append(aqpq5Shimmer)

            apq11Shimmer_list.append(apq11Shimmer)
            ddaShimmer_list.append(ddaShimmer)
            
            # add the formant data
            f1_mean_list.append(f1_mean)
            f2_mean_list.append(f2_mean)
            f3_mean_list.append(f3_mean)
            f4_mean_list.append(f4_mean)

            f1_median_list.append(f1_median)
            f2_median_list.append(f2_median)
            f3_median_list.append(f3_median)
            f4_median_list.append(f4_median)


            # //  +++++++++++++++++++++++++++++
            fdisp = []
            for i in range(len(f4_median_list)):
                fdisp.append((f4_median_list[i] - f1_median_list[i])/3)

            avgFormant = []
            for i in range(len(f4_median_list)):
                avgFormant.append((f1_median_list[i] + f2_median_list[i] + f3_median_list[i] + f4_median_list[i]) /4)

            mff = []
            for i in range(len(f4_median_list)):
                mff.append((f1_median_list[i] * f2_median_list[i] * f3_median_list[i] * f4_median_list[i]) **0.25)

            fitch_vtl = []
            for i in range(len(f4_median_list)):
                fitch_vtl.append(((1 * (35000 / (4 * f1_median_list[i]))) +
                                (3 * (35000 / (4 * f2_median_list[i]))) + 
                                (5 * (35000 / (4 * f3_median_list[i]))) + 
                                (7 * (35000 / (4 * f4_median_list[i])))) / 4 )

            xysum = []
            for i in range(len(f4_median_list)):
                xysum.append((0.5 * f1_median_list[i]) + 
                            (1.5 * f2_median_list[i]) + 
                            (2.5 * f3_median_list[i]) + 
                            (3.5 * f4_median_list[i]))

            xsquaredsum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
            delta_f = []
            for i in range(len(f4_median_list)):
                delta_f.append(xysum[i]/xsquaredsum)

            vtl_delta_f = []
            for i in range(len(f4_median_list)):
                vtl_delta_f.append(35000 / (2 * delta_f[i]))

            # subject = []
            # for i in range(len(f4_median_list)):
            #     subject.append(mark_)
            # //  +++++++++++++++++++++++++++++


            # Add the data to Pandas
            df = pd.DataFrame(np.column_stack([  mean_F0_list, sd_F0_list, hnr_list, 
                                            localJitter_list, localabsoluteJitter_list, rapJitter_list, 
                                            ppq5Jitter_list, ddpJitter_list, localShimmer_list, 
                                            localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, 
                                            apq11Shimmer_list, ddaShimmer_list, 
                                            f1_mean_list, f2_mean_list, f3_mean_list, f4_mean_list, 
                                            f1_median_list, f2_median_list, f3_median_list, f4_median_list,
                                            fdisp, avgFormant,mff,fitch_vtl,xysum,delta_f,vtl_delta_f
                                            ]),
                                            columns=[ 'meanF0Hz', 'stdevF0Hz', 'HNR', 
                                                        'localJitter', 'localabsoluteJitter', 'rapJitter', 
                                                        'ppq5Jitter', 'ddpJitter', 'localShimmer', 
                                                        'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 
                                                        'apq11Shimmer', 'ddaShimmer', 
                                                        'f1_mean', 'f2_mean', 'f3_mean', 'f4_mean', 
                                                        'f1_median', 'f2_median', 'f3_median', 'f4_median', 
                                                        'fdisp','avgFormant','mff','fitch_vtl','xysum','delta_f','vtl_delta_f'                                                    
                                                        ])

            X = np.array(df)
            predictions_ = loaded_model_vy_imp.predict(X)        
            # return jsonify({"show something":str('this is something')}) 
            return jsonify({"prediction":str(predictions_[0])}) 
    except:
        return jsonify({"prediction":str(2)})     



#  ----------------------------------------
#  predict_tremor_rest
#  ----------------------------------------
    
@app.route('/predict_tremor_rest', methods=['POST'])  
def predict_tremor_rest():
    try:
        if request.is_json:
            data = request.get_json()
            tStamp = []
            acX = []
            acY = []
            acZ = []
            agX = []
            agY = []
            agZ = []

            for i in data['recording']['recordedData']:
                tsC = i['ts']
                tStamp.append(tsC)
                acXC = i['data'][0]
                acYC = i['data'][1]
                acZC = i['data'][2]    
                acX.append(acXC)
                acY.append(acYC)
                acZ.append(acZC) 

                # agXC = i['data'][3]
                # agYC = i['data'][4]
                # agZC = i['data'][5]    
                # agX.append(agXC)
                # agY.append(agYC)
                # agZ.append(agZC) 

            tst = [item - tStamp[0] for item in tStamp]

            # ------------  handle the oversampling to 200 samples in 20 sec
            if len(acX) > 200:
                toBeSamp = 200
                # print('----> ' + str(filepath))
                acX, x1 = signal.resample(acX,toBeSamp,np.arange(len(acX)))  # resampled at 200
                acY, x1 = signal.resample(acY,toBeSamp,np.arange(len(acY)))  # resampled 
                acZ, x1 = signal.resample(acZ,toBeSamp,np.arange(len(acZ)))  # resampled 
                # agX, x1 = signal.resample(agX,toBeSamp,np.arange(len(agX)))  # resampled 
                # agY, x1 = signal.resample(agY,toBeSamp,np.arange(len(agY)))  # resampled
                # agZ, x1 = signal.resample(agZ,toBeSamp,np.arange(len(agZ)))  # resampled


            row = []
            for testsig in (acX,acY,acZ):
                testsig_filt = signal.sosfilt(sos, testsig)
                res = np.array(testsig_filt)
                fourier = fft(testsig_filt)
                fab = np.abs(fourier)[0:100]
                # ------------ 
                Esum = sum(np.square(fab))

                F1 = sum(np.square(fab[0:25]))
                F2 = sum(np.square(fab[25:50]))
                F3 = sum(np.square(fab[50:75]))
                F4 = sum(np.square(fab[75:80]))


                kur = kurtosis(testsig_filt, fisher=True)
                ske = skew(testsig_filt, bias=False)
                resdif = res[1:]-res[0:-1]
                Mobi = np.sqrt(np.var(resdif)/np.var(res))
                resdif2 = resdif[1:]-resdif[0:-1]
                compx = np.sqrt(np.var(resdif2)*np.var(res)/(np.var(resdif)*np.var(resdif)))
                
                # E1 = '%.5f'%(F1/Esum)
                E1 = '%.5f'%(np.std(testsig_filt))           
                E2 = '%.5f'%(np.mean(testsig_filt))                     
                E3 = '%.5f'%(kur)
                E4 = '%.5f'%(ske)
                E5 = '%.5f'%(Mobi)   
                E6 = '%.5f'%(compx)                              
                E7 = '%.5f'%(F1) 
                E8 = '%.5f'%(F2) 
                E9 = '%.5f'%(F3) 
                E10 = '%.5f'%(F4) 
                E11 = '%.5f'%(F2/Esum)
                E12 = '%.5f'%(F3/Esum)
                Samp, Phi1, Phi2 = EH.SampEn(res, m = 2, tau = 2)
                E13 = '%.5f'%(Samp[0])
                E14 = '%.5f'%(Samp[1])
                E15 = '%.5f'%(Samp[2])
                E16 = '%.5f'%(np.percentile(testsig_filt, 25))
                E17 = '%.5f'%(np.percentile(testsig_filt, 50))
                E18 = '%.5f'%(np.percentile(testsig_filt, 75))

                rowx = [E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15,E16,E17,E18]
                row = row + rowx
            
            toListofNumber = [float(x) for x in row]
            X = np.array([toListofNumber])
            predictions_ = loaded_model_tr.predict(X)        
            return jsonify({"prediction":str(predictions_[0])}) 
    # except:
    except Exception as e: 
        print(e)
        print('--------')
        print(rowx)
        print('--------')
        return jsonify({"prediction":str(2)}) 


#  ----------------------------------------
#  predict_tremor_post
#  ----------------------------------------
    
@app.route('/predict_tremor_post', methods=['POST'])  
def predict_tremor_post():
    try:
        if request.is_json:
            data = request.get_json()
            tStamp = []
            acX = []
            acY = []
            acZ = []
            agX = []
            agY = []
            agZ = []

            for i in data['recording']['recordedData']:
                tsC = i['ts']
                tStamp.append(tsC)
                acXC = i['data'][0]
                acYC = i['data'][1]
                acZC = i['data'][2]    
                acX.append(acXC)
                acY.append(acYC)
                acZ.append(acZC) 

                # agXC = i['data'][3]
                # agYC = i['data'][4]
                # agZC = i['data'][5]    
                # agX.append(agXC)
                # agY.append(agYC)
                # agZ.append(agZC) 

            tst = [item - tStamp[0] for item in tStamp]

            # ------------  handle the oversampling to 200 samples in 20 sec
            if len(acX) > 200:
                toBeSamp = 200
                # print('----> ' + str(filepath))
                acX, x1 = signal.resample(acX,toBeSamp,np.arange(len(acX)))  # resampled at 200
                acY, x1 = signal.resample(acY,toBeSamp,np.arange(len(acY)))  # resampled 
                acZ, x1 = signal.resample(acZ,toBeSamp,np.arange(len(acZ)))  # resampled 
                # agX, x1 = signal.resample(agX,toBeSamp,np.arange(len(agX)))  # resampled 
                # agY, x1 = signal.resample(agY,toBeSamp,np.arange(len(agY)))  # resampled
                # agZ, x1 = signal.resample(agZ,toBeSamp,np.arange(len(agZ)))  # resampled


            row = []
            # for testsig in (acX,acY,acZ,agX,agY,agZ):
            for testsig in (acX,acY,acZ):
                testsig_filt = signal.sosfilt(sos, testsig)
                res = np.array(testsig_filt)
                fourier = fft(testsig_filt)
                fab = np.abs(fourier)[0:100]
                # ------------ 
                Esum = sum(np.square(fab))

                F1 = sum(np.square(fab[0:25]))
                F2 = sum(np.square(fab[25:50]))
                F3 = sum(np.square(fab[50:75]))
                F4 = sum(np.square(fab[75:80]))


                kur = kurtosis(testsig_filt, fisher=True)
                ske = skew(testsig_filt, bias=False)
                resdif = res[1:]-res[0:-1]
                Mobi = np.sqrt(np.var(resdif)/np.var(res))
                resdif2 = resdif[1:]-resdif[0:-1]
                compx = np.sqrt(np.var(resdif2)*np.var(res)/(np.var(resdif)*np.var(resdif)))
                
                # E1 = '%.5f'%(F1/Esum)
                E1 = '%.5f'%(np.std(testsig_filt))           
                E2 = '%.5f'%(np.mean(testsig_filt))                     
                E3 = '%.5f'%(kur)
                E4 = '%.5f'%(ske)
                E5 = '%.5f'%(Mobi)   
                E6 = '%.5f'%(compx)                              
                E7 = '%.5f'%(F1) 
                E8 = '%.5f'%(F2) 
                E9 = '%.5f'%(F3) 
                E10 = '%.5f'%(F4) 
                E11 = '%.5f'%(F2/Esum)
                E12 = '%.5f'%(F3/Esum)
                Samp, Phi1, Phi2 = EH.SampEn(res, m = 2, tau = 2)
                E13 = '%.5f'%(Samp[0])
                E14 = '%.5f'%(Samp[1])
                E15 = '%.5f'%(Samp[2])
                E16 = '%.5f'%(np.percentile(testsig_filt, 25))
                E17 = '%.5f'%(np.percentile(testsig_filt, 50))
                E18 = '%.5f'%(np.percentile(testsig_filt, 75))

                rowx = [E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15,E16,E17,E18]
                row = row + rowx
            
            toListofNumber = [float(x) for x in row]
            X = np.array([toListofNumber])
            predictions_ = loaded_model_tp.predict(X)        
            return jsonify({"prediction":str(predictions_[0])}) 
    # except:
    except Exception as e: 
        print(e)
        print('--------')
        print(rowx)
        print('--------')
        return jsonify({"prediction":str(2)})   
    
