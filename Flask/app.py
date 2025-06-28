#importing from flask_request/dependencies
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os
app=Flask(__name__) # initializing a flask app

with open(r'C:/Users/johri/Credit_Card_Approval_Prediction/Flask/model.pkl','rb') as handle:
     model = pickle.load(handle)

@app.route('/') # route to display the home page
def home():
    return render_template('index.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('index1.html')
@app.route('/home',methods=['POST','GET'])
def my_home():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"]) # route to show the predictions in a web UI
def predict():
    try:
        # reading the inputs given by the user
        input_feature = [float(x) for x in request.form.values()]
        features_values = np.array([input_feature])
        feature_name = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',
                        'AMT_INCOME_TOTAL','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                    'NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS','paid_off','#_of_pastdues','no_loan']
        x=pd.DataFrame(features_values,columns=feature_name)

        # predictions using the loaded model file
        pred=model.predict(x)
        print(pred)
        if(pred==0):
            prediction = "Eligible"
        else:
            prediction = "Not Eligible"

        #prediction="Prediction is:"+str(pred)

        # showing the prediction results in a UI
        return render_template("result.html",prediction=prediction)
    except Exception as e:
        return f"Something went wrong: {e}"
    
if __name__=="__main__":

    # Running the app
    port=int(os.environ.get('PORT', 5000))
    app.run(port=port,debug=False,use_reloader=False)