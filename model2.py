import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('TrafficDataSet.csv')

data['Traffic Situation'] = data['Traffic Situation'].replace({'low': 1, 'normal': 2, 'high': 3, 'heavy': 4})
data['Day of the week'] = data['Day of the week'].replace({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})

data['hour'] = pd.to_datetime(data['Time']).dt.hour
data['minute'] = pd.to_datetime(data['Time']).dt.minute
data['AM/PM'] = (data['Time'].str.split().str[1] == 'PM').astype(int)

features = ['Day of the week', 'hour', 'minute', 'AM/PM']
targets = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']

scaler = StandardScaler()
scaler.fit(data[features])

X = data[features]
y = data['Traffic Situation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

models = {}
for target in targets:
    X = data[features]
    y = data[target]
    X_scaled = scaler.transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    models[target] = model

def predict_traffic(day, time_str):
    days_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    if day not in days_map:
        raise ValueError("Invalid day. Please enter a valid day (e.g., Monday)")

    try:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    except ValueError:
        raise ValueError("Invalid time format. Please use HH:MM:SS format")

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


while True:
    print("\nTraffic Flow Prediction System")
    print("-----------------------------")
    day = input("Enter day (e.g., Monday) or 'quit' to exit: ")

    if day.lower() == 'quit':
        break

    time_str = input("Enter time (HH:MM:SS): ")

    try:
        predictions, situation, total = predict_traffic(day, time_str)
        print("\nPredicted Traffic:")
        print("------------------")
        for vehicle_type, count in predictions.items():
            print(f"{vehicle_type}: {count}")
        print(f"\nTotal Vehicles: {total}")
        print(f"Traffic Situation: {situation}")
    except Exception as e:
        print(f"Error: {str(e)}")