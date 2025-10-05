from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model_path = "artifacts/models/model.pkl"
scaler_path = "artifacts/processed/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html',prediction=None)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        healthcare_cost = float(request.form["healthcare_costs"])
        tumer_size = float(request.form["tumor_size"])
        treatment_type = float(request.form["treatment_type"])
        diabeties  = int(request.form["diabetes"])
        mortality_rate = float(request.form["mortality_rate"])

        input_formodel = np.array([[healthcare_cost, tumer_size, treatment_type, diabeties, mortality_rate]])

        scaled_input = scaler.transform(input_formodel)

        prediction = model.predict(scaled_input)[0]

        return render_template('index.html', prediction = prediction)
    except Exception as e:
        return str(e)
    
if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)