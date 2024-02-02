from flask import Flask, jsonify
from flask import render_template

app = Flask(__name__)


@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})

@app.route('/')
def saysomething():
    return ("now what -------------")

