# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)

plt.plot(
    [5, 4, 3], 
    [100, 200, 300] 
)
plt.title('Some Title')
plt.xlabel('Year')
plt.ylabel('Some measurements')

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
current_time=current_time.split(":")
current_time=''.join(current_time)
file_location='static/images/'+str(current_time)+'.png'
plt.savefig(file_location)
# print(current_time,type(current_time),type(str(current_time)))

data = pd.read_csv('Preprocessed_data.csv')

print(data.shape)

X=data[['Accel_sec','Range_Km','TopSpeed_KmH','Efficiency_WhKm']]
y=data['PriceEuro']

# print(X)
# print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_train, y_train)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Time = float(request.form['Time'])
            V1 =  (request.form['V1'])
            V2 =  float(request.form['V2'])
            V3 =  float(request.form['V3'])
            V4 =  float(request.form['V4'])
            V5 =  float(request.form['V5'])
            V6 =  (request.form['V6'])
            V7 =  (request.form['V7'])
            V8 =  (request.form['V8'])
            V9 =  (request.form['V9'])
            V10 =  (request.form['V10'])
            V11 =  (request.form['V11'])
            V12 =  float(request.form['V12'])
            V13 =  float(request.form['V13'])
            
            # Now we will create the list inorder to pass the value to the model
            pred_args = [V2, V3, V4, V5]
            # pred_args=[V2,V4,V3,V5]
            pred_agrs_arr = np.array(pred_args)
            pred_agrs_arr = pred_agrs_arr.reshape(1,-1)
            model_prediction = lr.predict(pred_agrs_arr)
            model_prediction = int(model_prediction)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction,file_location=file_location)



# @app.route("/Image_Processing")
# def image():
#     # return the homepage
#     return render_template("image.html")



if __name__ == '__main__':
    app.run(debug=True)
    
    
    
