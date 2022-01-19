import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import sys
import re
import joblib
import socket
import json
import numpy as np
import pandas as pd

sys.path.append('.')

from executor.model_inferrer import ModelInferrer

model = ModelInferrer()

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

#this is how we are getting the file that the user uploads. 
#then we are setting the path that we want to save it so we can use it later for predictions
@app.route("/", methods=['GET', 'POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = ("app/static/files/bank.csv")
        uploaded_file.save(file_path)
    return redirect(url_for('downloadFile'))

#now we are reading the file, make predictions with our model and save the predictions.
#then we are sending the CSV with the predictions to the user as attachement 
@app.route('/download')
def downloadFile ():
    path = "app/static/files/bank.csv"
    predictions=model.infer(pd.read_csv(path, delimiter=';'))
    predictions.to_csv('app/static/files/predictions.csv',index=False)
    return send_file("static/files/predictions.csv", as_attachment=True)

if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True ,port=8080)

