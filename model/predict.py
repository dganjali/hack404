import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import json

# Load data with correct date column
df = pd.read_csv("../data/shelter_occupancy.csv", parse_dates=["OCCUPANCY_DATE"])

# Sort by shelter and date
df = df.sort_values(["FACILITY_NAME", "OCCUPANCY_DATE"])

# Create lag feature: previous day occupancy per shelter
df['prev_occupancy'] = df.groupby('FACILITY_NAME')['OCCUPANCY'].shift(1)

# Drop rows with missing lag
df = df.dropna(subset=['prev_occupancy'])

# Add date features
df['dow'] = df['OCCUPANCY_DATE'].dt.weekday
df['month'] = df['OCCUPANCY_DATE'].dt.month

# Features and target
X = df[['dow', 'month', 'prev_occupancy']]
y = df['OCCUPANCY']

# Train model (you can split or use all data)
model = RandomForestRegressor()
model.fit(X, y)

# Predict for today:
today = pd.Timestamp.today()
dow = today.weekday()
month = today.month

# For each shelter, get yesterday's occupancy
latest = df.groupby('FACILITY_NAME').last().reset_index()
latest['dow'] = dow
latest['month'] = month

X_pred = latest[['dow', 'month', 'prev_occupancy']]
preds = model.predict(X_pred)

# Create predictions JSON
predictions = []
for name, pred in zip(latest['FACILITY_NAME'], preds):
    predictions.append({"name": name, "predicted_influx": int(pred)})

with open('../data/predictions.json', 'w') as f:
    json.dump(predictions, f)
