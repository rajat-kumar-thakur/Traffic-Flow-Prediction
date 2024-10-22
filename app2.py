from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

app = Flask(__name__)

# Load and preprocess the data
data = pd.read_csv('TrafficDataSet.csv')

# Replace categorical values with numerical
data['Traffic Situation'] = data['Traffic Situation'].replace({'low': 1, 'normal': 2, 'high': 3, 'heavy': 4})
data['Day of the week'] = data['Day of the week'].replace(
    {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})

# Add time features
data['hour'] = pd.to_datetime(data['Time']).dt.hour
data['minute'] = pd.to_datetime(data['Time']).dt.minute
data['AM/PM'] = (data['Time'].str.split().str[1] == 'PM').astype(int)

# Features and target columns
features = ['Day of the week', 'hour', 'minute', 'AM/PM']
targets = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']

# StandardScaler for feature scaling
scaler = StandardScaler()
scaler.fit(data[features])

# Train RandomForest models for each vehicle type
models = {}
for target in targets:
    X = data[features]
    y = data[target]
    X_scaled = scaler.transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    models[target] = model


# Prediction function
def predict_traffic(day, time_str):
    days_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

    try:
        time_obj = datetime.strptime(time_str, '%H:%M')
    except ValueError:
        raise ValueError("Invalid time format. Please use HH:MM format")

    hour = time_obj.hour
    minute = time_obj.minute
    am_pm = 1 if hour >= 12 else 0
    hour = hour % 12 if hour % 12 != 0 else 12

    input_data = pd.DataFrame([[days_map[day], hour, minute, am_pm]], columns=features)
    input_scaled = scaler.transform(input_data)

    predictions = {}
    for vehicle_type in models:
        pred = max(0, round(models[vehicle_type].predict(input_scaled)[0]))
        predictions[vehicle_type] = pred

    total_vehicles = sum(predictions.values())
    if total_vehicles < 50:
        situation = "Low"
    elif total_vehicles < 100:
        situation = "Normal"
    elif total_vehicles < 150:
        situation = "High"
    else:
        situation = "Heavy"

    return predictions, situation, total_vehicles


# Flask route for the homepage
@app.route('/')
def index():
    return render_template('index2.html')


# Flask route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    day = request.form.get('day')
    time = request.form.get('time')

    try:
        predictions, situation, total = predict_traffic(day, time)
        return render_template('index2.html', predictions=predictions, situation=situation, total=total, day=day,
                               time=time)
    except Exception as e:
        return render_template('index2.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
