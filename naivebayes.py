import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

try:
    data = pd.read_csv('TrafficDataSet.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'TrafficDataSet"
          "..csv' file not found. Please ensure the file is in the current directory.")
    exit()


situation_mapping = {'low': 0, 'normal': 1, 'heavy': 2, 'high': 3}
data['Traffic Situation'] = data['Traffic Situation'].map(situation_mapping)

day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
               'Friday': 5, 'Saturday': 6, 'Sunday': 7}
data['Day of the week'] = data['Day of the week'].map(day_mapping)

data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p')
data['hour'] = data['Time'].dt.hour
data['minute'] = data['Time'].dt.minute

data = data.drop(columns=['Date', 'Time'], axis=1)



X = data[['Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'hour', 'minute']]
y = data['Traffic Situation']

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

print(f"\nTraining samples: {train_X.shape[0]}, Testing samples: {test_X.shape[0]}")

sc = StandardScaler()
train_X_scaled = sc.fit_transform(train_X)
test_X_scaled = sc.transform(test_X)

gnb = GaussianNB()
gnb.fit(train_X_scaled, train_y)

print("\nGaussian Naive Bayes classifier trained successfully.")

gnb_pred = gnb.predict(test_X_scaled)

accuracy = accuracy_score(test_y, gnb_pred)
print(f"\nNaive Bayes Classifier Accuracy: {accuracy:.4f}")



def predict_traffic_situation(model, scaler, day_mapping, reverse_situation_mapping):
    print("\n--- Traffic Situation Prediction ---")

    day_input = input("Enter the day of the week (e.g., Monday): ").strip()
    day_num = day_mapping.get(day_input.capitalize(), None)

    if day_num is None:
        print("Invalid day entered. Please try again.")
        return

    time_input = input("Enter the time in HH:MM format (e.g., 08:30): ").strip()
    try:
        time_parsed = pd.to_datetime(time_input, format='%H:%M')
        hour = time_parsed.hour
        minute = time_parsed.minute
    except ValueError:
        print("Invalid time format entered. Please enter time in HH:MM format (e.g., 08:30).")
        return

    try:
        car_count = int(input("Enter the number of cars: ").strip())
        bike_count = int(input("Enter the number of bikes: ").strip())
        bus_count = int(input("Enter the number of buses: ").strip())
        truck_count = int(input("Enter the number of trucks: ").strip())
    except ValueError:
        print("Invalid input for vehicle counts. Please enter integer values.")
        return

    total = car_count + bike_count + bus_count + truck_count

    user_features = [[day_num, car_count, bike_count, bus_count, truck_count, total, hour, minute]]

    user_features_scaled = scaler.transform(user_features)

    probabilities = model.predict_proba(user_features_scaled)[0]

    print("\nAverage Prediction (Probabilities):")
    for idx, prob in enumerate(probabilities):
        situation = reverse_situation_mapping[idx]
        print(f"Probability of '{situation}': {prob:.4f}")

    predicted_class_index = probabilities.argmax()
    predicted_situation = reverse_situation_mapping[predicted_class_index]

    print(f"\nVerdict: Predicted Traffic Situation is '{predicted_situation}'")

reverse_situation_mapping = {v: k for k, v in situation_mapping.items()}

predict_traffic_situation(gnb, sc, day_mapping, reverse_situation_mapping)