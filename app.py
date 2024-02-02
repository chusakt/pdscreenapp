from flask import Flask, jsonify
from flask import render_template
from flask import request, Response, send_file, redirect, safe_join, abort

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
import requests
import gzip
import io
from io import BytesIO
import threading
import zipfile
from zipfile import ZipFile 
import urllib.parse


import numpy as np
from numpy import mean, sqrt, square, arange
from numpy import genfromtxt

from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.signal import welch, hann
import pandas as pd



app = Flask(__name__)


@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})

@app.route('/')
def saysomething():
    return ("now what -------------")

