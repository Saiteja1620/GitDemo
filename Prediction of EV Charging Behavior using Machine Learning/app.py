# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import sqlite3
import numpy as np
import pandas as pd

app = Flask(__name__)

data = pd.read_csv('ev_data_processed.csv')

print(data.shape)

X = data.iloc[:, 2:23]
y = data.iloc[:,1]

print(X)
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            Time = float(request.form['Time'])
            V1 =  float(request.form['V1'])
            V2 =  float(request.form['V2'])
            V3 =  float(request.form['V3'])
            V4 =  float(request.form['V4'])
            V5 =  float(request.form['V5'])
            V6 =  float(request.form['V6'])
            V7 =  float(request.form['V7'])
            V8 =  float(request.form['V8'])
            V9 =  float(request.form['V9'])
            V10 =  float(request.form['V10'])
            V11 =  float(request.form['V11'])
            V12 =  float(request.form['V12'])
            V13 =  float(request.form['V13'])
            V14 =  float(request.form['V14'])
            V15 =  float(request.form['V15'])
            V16 =  float(request.form['V16'])
            V17 =  float(request.form['V17'])
            V18 =  float(request.form['V18'])
            V19 =  float(request.form['V19'])
            V20 =  float(request.form['V20'])
            
            
            # Now we will create the list inorder to pass the value to the model
            pred_args = [Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, \
                         V12, V13, V14, V15, V16, V17, V18, V19,V20]
            pred_agrs_arr = np.array(pred_args)
            pred_agrs_arr = pred_agrs_arr.reshape(1,-1)
            model_prediction = RF.predict(pred_agrs_arr)
            model_prediction = int(model_prediction)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)



@app.route("/Image_Processing")
def image():
    # return the homepage
    return render_template("image.html")



if __name__ == '__main__':
    app.run(debug=True)
    
    
    
