import os, json
import pandas as pd
import requests
from flask import Flask, render_template

app = Flask(__name__)

# @app.route('/')
# @app.route('/index')
@app.route('/prediction', methods=['GET','POST'])
def predict():
    
    # load image and turn it into a json list
    predict_df = pd.read_csv("predictions.csv")
    predict_df = predict_df.to_json()
    
    # capture request response
    response = app.response_class(
    response = json.dumps(predict_df),
    status=200,
    mimetype='application/json'
    )
    
    return response

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)
