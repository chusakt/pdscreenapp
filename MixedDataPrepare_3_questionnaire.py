# -*- coding: utf-8 -*-
"""
Created on 2022 03 21

@author: boon
"""

#%%
import os
from os import walk
from glob import glob
import pathlib
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np

# check filled up sheets
# -------------------------------------------------------------
# -------------------------------------------------------------
# rawdatfolder = 'PD_fullmark_nof8_C'                   # <---------
# diag = 'pd'                                         # <---------
# TargetbaseDir = "./DATA/Prepare_fullnof8_C/questionPD"  # <---------
# baseDir = "./DATA/"+rawdatfolder
# fileNameToCheck = "--"              # <---------
# matchto = "*/*/*/*T1*"                          # <---------
# -------------------------------------------------------------

# -------------------------------------------------------------
rawdatfolder = 'Control_fullmark_nof8_C'                   # <---------
diag = 'pd'                                         # <---------
TargetbaseDir = "./DATA/Prepare_fullnof8_C/questionCT"  # <---------
baseDir = "./DATA/"+rawdatfolder
fileNameToCheck = "--"              # <---------
matchto = "*/*/*/*T1*"                          # <---------



# fileNameToCheck = "Ahh1"
# matchto = "*/*/*/*T4*"                         
# fileNameToCheck4 = "YaiParLarn"
# fileNameToCheck5 = "pdwalk"
# fileNameToCheck6 = "PosturalStability"
# fileNameToCheck7 = "figureOfEightDraw"

fileAnno = baseDir


isExist = os.path.exists(TargetbaseDir)
# if isExist:
#     shutil.rmtree(TargetbaseDir)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(TargetbaseDir)

# to store file names
res = []
res2 = []
# construct path object
d = pathlib.Path(fileAnno)

res3 = []
for (dir_path, dir_names, file_names) in walk(fileAnno):
    res3.extend(dir_names)
    # don't look inside any subdirectory
    break    
print("Subject Count: "+str(len(res3)))


# iterate directory -- list it
for entry in d.iterdir():
    # check if it a dir
    # if entry.is_dir():
    #     res.append(entry)
    for f in entry.iterdir():
        # check if it a dir
        if f.is_dir():
            res2.append(f)        
# print(res2)
print("Test Count (subfolders): "+str(len(res2)))

# select only T2 test
res = []
for file_names in res2:
    if pathlib.Path(file_names).match(matchto):
        res.append(file_names)
# print(res)
print(matchto + " test Count: "+str(len(res)))

# check if in res has good file recorded digitally
# select only xlsx files
resTremor = []
for fd in res:
    for entry in fd.iterdir():
        if entry.is_file(): #check in case of a file
            if pathlib.Path(entry).match('*.xlsx'): 
                resTremor.append(entry)        
   

# manage the record in sub Date
for fd in res:
    for entry in fd.iterdir():
        if entry.is_dir(): #check in case of a dir and select only digitally 
            # print(entry)
            for path in os.listdir(entry):
                # check if current path is a file
                if os.path.isfile(os.path.join(entry, path)):
                    # print (path+'---pppppppppppp')
                    # print(pathlib.Path(entry, path))
                    if pathlib.Path(entry, path).match('*.xlsx'): 
                        resTremor.append(pathlib.Path(entry, path))

# print(resTremor)
print(fileNameToCheck + " Count: "+str(len(resTremor)))
print("----------------")
# ------------------------
# copy all tremor json to TargetbaseDir
for idx, fileT in enumerate(resTremor):
    mrk = fileT.__str__().find(rawdatfolder)
    mrk2 = fileT.__str__().find('\T',len(rawdatfolder))
    getfn = fileT.__str__()[mrk+len(rawdatfolder)+1:mrk2]
    print(getfn)
    if (os.path.exists(TargetbaseDir.__str__()+'/'+getfn+'_1.xlsx')) == False:
        shutil.copy(fileT, (TargetbaseDir.__str__()+'/'+getfn+'_1.xlsx'))
    elif (os.path.exists(TargetbaseDir.__str__()+'/'+getfn+'_2.xlsx')) == False:
        shutil.copy(fileT, (TargetbaseDir.__str__()+'/'+getfn+'_2.xlsx'))
    elif (os.path.exists(TargetbaseDir.__str__()+'/'+getfn+'_3.xlsx')) == False:
        shutil.copy(fileT, (TargetbaseDir.__str__()+'/'+getfn+'_3.xlsx'))
    elif (os.path.exists(TargetbaseDir.__str__()+'/'+getfn+'_4.xlsx')) == False:
        shutil.copy(fileT, (TargetbaseDir.__str__()+'/'+getfn+'_4.xlsx'))
    
    # print('Copied')


# # ------------------------
# # add key:value to json
# res3 = []
# for (dir_path, dir_names, file_names) in walk(TargetbaseDir):
#     res3.extend(file_names)
# # print("new folder count file: "+str(len(res3)))
# files_       = glob(os.path.join(TargetbaseDir, "*.json"))
# for i in files_:
#     with open(i, encoding="utf8") as json_file:
#         json_decoded = json.load(json_file)
#     json_decoded['diag'] = diag
#     with open(i, 'w') as json_file:
#         json.dump(json_decoded, json_file)

