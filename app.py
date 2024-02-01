from flask import Flask, jsonify 
from flask import Flask, render_template, request, jsonify 
 
 
@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    return jsonify({"uuid":uuid})
 
 
if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)