import os, json
from flask import Flask, render_template

import sys
sys.path.append('.')

app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/data')

@app.route(APP_ROOT, methods=['GET','POST'])
def infer():
    
    # load image and turn it into a json list
    data = {'age': [30, 33],
            'job': ["unemployed", "services"],
            "marital": ["married", "married"],
            "education": ["primary", "secondary"],
            "default": ["no", "no"],
            "balance": [1787, 4789],
            "housing": ["no", "yes"],
            "loan": ["no", "yes"],
            "contact": ["cellular", "cellular"],
            "day": [19, 11],
            "month": ["oct", "may"],
            "duration": [79, 220],
            "campaign": [1, 1],
            "pdays": [-1, 339],
            "previous": [0, 4],
            "poutcome":["unknown", "failure"]
    }
    
    # capture request response
    response = app.response_class(
    response = json.dumps(data),
    status=200,
    mimetype='application/json'
    )

    return response

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)


