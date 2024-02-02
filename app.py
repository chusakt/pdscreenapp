from flask import Flask, jsonify 
from flask import Flask, render_template, request, jsonify 

import time, threading, csv, random, json
from datetime import datetime, timedelta
import unicodedata
import numpy as np
from numpy import mean, sqrt, square, arange
from   numpy import genfromtxt
from flask import request, Response, send_file, redirect, safe_join, abort
# from scipy.signal import butter, lfilter, freqz, filtfilt
# from scipy.signal import welch, hann

import pandas as pd

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
# import matplotlib.dates as mdates
# from matplotlib.patches import Ellipse
# import matplotlib.transforms as transforms
# import PIL

import re
import os
import shutil
import time
import random
import datetime
import base64
from decimal import Decimal
import requests
import gzip
import io
from io import BytesIO
import threading
import zipfile
from zipfile import ZipFile 
import urllib.parse
import logging
ENCODING = 'utf-8'



app = Flask(__name__)



def movingaverage(testsequence, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(testsequence, window, 'same')



@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})

@app.route('/')
def saysomething():
    return ("now what -------------")
 
 
@app.route('/walking6min', methods=['POST'])  
def walking6min():
    # def walking6min(DOWNSMP_OF_RAW=2): #def postRR(DOWNSMP_OF_RAW=4):
    smoothfactor            = 3    # filter smooth average
    TimeFunctionCalled = ((datetime.datetime.now()+ timedelta(hours=7)).strftime("%H%M%S"))
    if request.is_json:
        return ('... received Json ... ')
        # req = request.get_json()


#         inputSecretCode = req['secretCode']
#         # if inputSecretCode != secretSoSecretCodeHere:
#         #     jsonify({"Response":"Incorrect Passcode" })  
#         b64	        =  req['data'] #-- on test, use local base64 txt file
                
#         myTokenPointTo =  LINE_ACCESS_TOKEN_TESTER
#         datafolder = './walk6min/'
#         tempfolder = './walk6mintemp/'
                
# ## xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#         patient_    = "w6"
#         b64	        =  req['data']
# ## ---------------
#         deviceType_    = "N/A"
#         if ("deviceType" in req):
#             deviceType_   = req['deviceType']
#             if deviceType_ == "echem2022":
#                 patient_ = "eC"
#         else:
#             deviceType_    = "N/A"
# ## ---------------
#         patientName_    = ""
#         if ("patientName" in req):
#             patientName_   = req['patientName']
#         else:
#             patientName_    = ""

#         nowStamp = ((datetime.datetime.now()+ timedelta(hours=7)).strftime("%Y%m%d_%H%M%S"))
#         TimeFunctionCalled = ((datetime.datetime.now()+ timedelta(hours=7)).strftime("%H%M%S"))
#         # filename_head = username_+'_'+nowStamp
#         filename_head = patient_+'_'+nowStamp
#         filename_csv = datafolder+'_'+filename_head+'_' +str(patientName_)+".csv"
#         filename_zip = datafolder+filename_head+'_'+str(patientName_) +".zip" 
#         # fileToWriteTR = datafolder+'_'+filename_head +"TR.csv" 
#         # fileToWriteWT = datafolder+'_'+filename_head +"WT.csv"      # DEGREE OF "WRIST" SENSOR AS CSV

#         # make temp dir for each incoming data set
#         tempfolder_sub = tempfolder+filename_head+'_' +str(patientName_)+'/'
#         if not os.path.exists(tempfolder_sub):
#             os.makedirs(tempfolder_sub)
        
#         try:
#             with open(filename_zip, "wb") as fh:
#                 fh.write(base64.decodebytes(b64.encode('ascii')))
#                 fileRsize = os.path.getsize(fh.name)
#                 fh.close()
#             # del b64
            
#         except OSError as e:
#             logging.exception("message")
#             return jsonify({"Response":"failed with base64 decode or read file" }) 

#         # -- Process zip file, 1 unzip, 2 process
#         # Create a ZipFile Object and load sample.zip in it
#         with ZipFile(filename_zip, 'r') as zipObj:
#             # Extract all the contents of zip file in different directory
#             zipObj.extractall(tempfolder_sub)

#         # Process *************** -- ***************** -- *****************
#         try: 
#             data_path = tempfolder_sub+"data.csv"
#             info_path = tempfolder_sub+"info.csv"
#             # read info
#             reader = csv.reader(info_path, delimiter=',')
#             df = pd.read_csv(info_path)
#             sN = df['Name'] 
#             sH = df['Height'] 
#             sA = df['Age']
#             sSL = df['Sensor Position'] 
#             sWalkDistRecord = df['Wdistance'] 
#             sStrideRecord = df['Stride'] 
            
#             subName = sN[0]
#             subHeight = sH[0]
#             subAge = sA[0]
#             subSensorLocation = sSL[0]
#             subWalkRecord = sWalkDistRecord[0]
#             subStrideRecord = sStrideRecord[0]
#             subStrideRecord = 0          # <---------------------
            

            
            
#             # DisEst[m] = 335.923-(1.408*Age)-(0.617*Hight[cm])+(0.605*DisFrmEstStride[m])


            
#             # fix error of csv
#             data = [0,1,1,1]
#             with open(data_path, 'r') as f:
#                 reader = csv.reader(f, delimiter=',')
#                 headers1 = next(reader)
#                 for row in reader:
#                     # print (len(row))
#                     row = list(np.float_(row))
#                     if (len(row) == 4):
#                         data = np.vstack([data, row])
        
#             # START PROCESS DATA
#             dataOnly = data[2:,2]
#             if len(dataOnly) > 3600:
#                 dataOnly = dataOnly[:3600]

#             fs = 10
#             # Only specific sensor is analysed
#             read_matrix                 = dataOnly
#             t                           = np.linspace(0, read_matrix.shape[0], read_matrix.shape[0], endpoint=False)
#             timing_scale_full           = t*(1/fs) # convert time axis to Minute

#             read_matrix_SM = movingaverage(read_matrix,smoothfactor)
#             read_matrix_SM_adj = read_matrix_SM - mean(read_matrix_SM)

#             y= np.sign(read_matrix_SM_adj)
#             z= y[1:]-y[0:-1]
#             downward_zerocross = (z==-2)
#             downward_zerocross_location = downward_zerocross*1
    

#             # # # -- cal walk Distance -- [method 2]
#             bothLegStepCount = 2*sum(downward_zerocross_location)
#             step_per_2sec = 2*bothLegStepCount/timing_scale_full[-1]
#             if step_per_2sec < 3.5:
#                 stride = subHeight/3
#             elif step_per_2sec >= 3.5 and step_per_2sec < 4.5:
#                 stride = subHeight/2.5
#             elif step_per_2sec >= 4.5 and step_per_2sec < 5.5:
#                 stride = subHeight/2
#             elif step_per_2sec >= 5.5:
#                 stride = subHeight/1.5
#             DisFrmEstStride = stride*bothLegStepCount
#             DisEst = 335.923-(1.408*subAge)-(0.617*subHeight)+(0.605*DisFrmEstStride/100)  ## <-----------
#             d_estimate_ = format(DisEst, ".2f")
    
#             stride_fromUser = subStrideRecord*100
#             distance_fromUserStride = stride_fromUser*bothLegStepCount
#             # dse_  =  format(stride*bothLegStepCount/100, ".2f")

#             Nam_ = str(subName)
#             Hei_ = format(subHeight, ".2f")
            
#             # Loc_ = str(subSensorLocation)
#             stp_ = str(bothLegStepCount)
#             # s2s_ = format(step_per_2sec, ".2f")
#             # ste_ = format(stride, ".2f")
#             stu_ = format(stride_fromUser, ".2f")
#             dsu_ = format(distance_fromUserStride/100, ".2f")
            
#             # dsr_ = format(subWalkRecord, ".2f")

# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                        
#             # -- get DATE with downsampled
#             # time_list = []
#             # for count, row in enumerate(reader, start=2):
#             #     if count % (DOWNSMP_OF_RAW*DOWNSMP_OF_STP2) == 0:
#             #         time_list.append(int(row[0][0:-3]))

#             # dates=[datetime.datetime.fromtimestamp(ts)+ timedelta(hours=7) for ts in time_list]
#             # del time_list
            
#         except OSError as e:
#             logging.exception("message")
#             return jsonify({"Response":"failed during being processed" })  
                
#         # -- LINE NOTIFICATION --**
#         # message = "6m-Walk build20B.\nFilesize: "+str(fileRsize)+"\nName: "+str(patientName_)
#         # message = "6m-Walk build21.\nName: "+str(patientName_)+"\nDistance: "+dsu_+" m."
#         message = "6m-Walk build21.\nName: "+str(patientName_)+"\nDistance: "+d_estimate_+" m."
#         if deviceType_ == "echem2022":
#             message = "eChem upload for: "+str(patientName_)
            
#         msg = urllib.parse.urlencode({"message":message})
#         LINE_ACCESS_TOKEN_toUse = LINE_ACCESS_TOKEN_WALK
#         LINE_HEADERS = {'Content-Type':'application/x-www-form-urlencoded',"Authorization":"Bearer "+LINE_ACCESS_TOKEN_toUse}
#         session = requests.Session()
#         a=session.post(urlline, headers=LINE_HEADERS, data=msg)
#         # -- LINE NOTIFICATION --**
        
#         return jsonify({"Response":d_estimate_ })  
    
#     else:
#         print ('ERROR: DATA NOT JSON')

#         # -- LINE NOTIFICATION --**
#         message = "Walking data upload ERROR"
#         msg = urllib.parse.urlencode({"message":message})
#         LINE_ACCESS_TOKEN_toUse = LINE_ACCESS_TOKEN_WALK
#         LINE_HEADERS = {'Content-Type':'application/x-www-form-urlencoded',"Authorization":"Bearer "+LINE_ACCESS_TOKEN_toUse}
#         session = requests.Session()
#         a=session.post(urlline, headers=LINE_HEADERS, data=msg)
#         # -- LINE NOTIFICATION --**
        
#         return jsonify({"Response":"No Json" })
