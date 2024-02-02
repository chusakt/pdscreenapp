from flask import Flask, jsonify 
from flask import Flask, render_template, request, jsonify 

app = Flask(__name__)
 
@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})

@app.route('/')
def saysomething():
    return ("want to make change")
 
 
