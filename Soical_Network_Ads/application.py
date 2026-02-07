import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from flask import Flask,render_template,request,jsonify 



# Import models 
girdModel = pickle.load(open("Models/grid.pkl","rb"))
standard_scaler = pickle.load(open("Models/scaler.pkl","rb"))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predication",methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        Gender = request.form.get("gender")
        Age = float(request.form.get("age"))
        Salary = float(request.form.get("salary"))

        gender_numeric = 1 if Gender == 'Male' else 0
        new_data_scaled = standard_scaler.transform([[gender_numeric, Age, Salary]])

# DEBUG: Look at your terminal (not the browser) when you click Predict
        print(f"DEBUG - Raw Input: {gender_numeric}, {Age}, {Salary}")
        print(f"DEBUG - Scaled Input: {new_data_scaled}")

        prediction = girdModel.predict(new_data_scaled)

        # new_data_scaled = standard_scaler.transform([[gender_numeric,Age,Salary]])
        # prediction = girdModel.predict(new_data_scaled)
        result_text = "Likely to Purchase" if prediction[0] == 1 else "Unlikely to Purchase"

        return render_template("predication.html",prediction_text=result_text)

    return render_template("predication.html")





if __name__  == "__main__":
    app.run(debug=True)
