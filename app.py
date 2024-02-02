from flask import Flask, jsonify 
from flask import Flask, render_template, request, jsonify 
# from flask import request, Response, send_file, redirect, safe_join, abort

app = Flask(__name__)

@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})

@app.route('/')
def saysomething():
    return ("now what -------------")
 
 
# @app.route('/walking6min')  
# def walking6min():
#     return (" in walking function ")
