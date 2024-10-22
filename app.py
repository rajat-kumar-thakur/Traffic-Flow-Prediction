from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('TrafficDataSet.csv')
data['Traffic Situation'] = data['Traffic Situation'].replace({'low': 1, 'normal': 2, 'high': 3, 'heavy': 4})
data['Day of the week'] = data['Day of the week'].replace(
    {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})

data['hour'] = pd.to_datetime(data['Time']).dt.hour
data['minute'] = pd.to_datetime(data['Time']).dt.minute
data['AM/PM'] = (data['Time'].str.split().str[1] == 'PM').astype(int)
data = data.drop(columns=['Time'], axis=1)

# Prepare data for training
X = data[
    ['Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'hour', 'minute', 'AM/PM']]
y = data['Traffic Situation'].values

# Split and scale data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

# Train model
rfc = RandomForestClassifier(random_state=0)
rfc.fit(train_X, train_y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        day = data['day']
        time = data['time']

        days_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
        day_num = days_map[day]

        time_obj = datetime.strptime(time, '%H:%M:%S')
        hour = time_obj.hour
        minute = time_obj.minute
        am_pm = 1 if hour >= 12 else 0
        hour = hour % 12

        counts = {
            'car_count': int(data['car_count']),
            'bike_count': int(data['bike_count']),
            'bus_count': int(data['bus_count']),
            'truck_count': int(data['truck_count'])
        }

        total_vehicles = sum(counts.values())

        input_data = pd.DataFrame(
            [[0, day_num, counts['car_count'], counts['bike_count'],
              counts['bus_count'], counts['truck_count'],
              total_vehicles, hour, minute, am_pm]],
            columns=['Date', 'Day of the week', 'CarCount', 'BikeCount',
                     'BusCount', 'TruckCount', 'Total', 'hour', 'minute', 'AM/PM']
        )

        input_data_scaled = sc.transform(input_data)
        prediction = rfc.predict(input_data_scaled)[0]

        traffic_situation = {1: 'low', 2: 'normal', 3: 'high', 4: 'heavy'}

        return jsonify({
            'prediction': traffic_situation[prediction],
            'prediction_code': int(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)