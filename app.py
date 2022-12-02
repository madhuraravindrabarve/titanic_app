import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
import json
from joblib import load


app = Flask(__name__)
model = load('pred_survival_titanic.sav')


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])       
def predict():

    int_features = [str(x) for x in request.form.values()]
    print("input",int_features)
    sex = int_features[0]
    sex = 0 if sex != 'male' else 1
    print("sex",sex)

    pclass = int_features[1]
    print("pclass",pclass)

    X = [[pclass, int(sex != 'male')]]
    final_features = X
    prediction = model.predict(final_features)
    if prediction == 0:
        prediction = "Did not survived"
    elif prediction == 1:
        prediction = "survived!"

    print(prediction)

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="The passanger {}".format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)