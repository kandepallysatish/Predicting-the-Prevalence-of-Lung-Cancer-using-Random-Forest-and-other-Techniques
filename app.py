import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
import pickle

from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__) #Initialize the flask App


TrainedModel = pickle.load(open('TrainedRandomForest.pkl','rb'))

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')


@app.route('/login')
def login():
    return render_template('login.html')
    

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
     
    
    final_features = [np.array(int_feature, dtype='float64')]
     
    result=TrainedModel.predict(final_features)
    print("result:", result)
    if result == 1:
            result = "Lung Cancer not Detected"
    else:
        result = 'Lung Cancer Detected'
    return render_template('prediction.html', prediction_text= result)


@app.route('/performance')
def performance():
    return render_template('performance.html')   
    
if __name__ == "__main__":
    app.run()
