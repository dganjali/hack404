from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import numpy as np
from datetime import datetime
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../backend/trained_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), '../data/real_features.json')

try:
    model = joblib.load(model_path)
    print(f"✅ Loaded trained model from {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Load real features
try:
    with open(features_path, 'r') as f:
        real_features = json.load(f)
    print(f"✅ Loaded {len(real_features)} real feature records")
except Exception as e:
    print(f"❌ Failed to load features: {e}")
    real_features = []

class PredictRequest(BaseModel):
    features: Dict[str, Any] = {}

def create_current_features():
    """Create features for current date based on real data patterns"""
    if not real_features:
        return [0] * 19  # Default features
    
    current_date = datetime.now()
    year = current_date.year
    month = current_date.month
    day_of_week = current_date.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_spring = 1 if month in [3, 4, 5] else 0
    is_fall = 1 if month in [9, 10, 11] else 0
    
    # Use historical averages for the rest
    avg_occupancy = np.mean([f['avg_occupancy'] for f in real_features])
    max_occupancy = np.mean([f['max_occupancy'] for f in real_features])
    min_occupancy = np.mean([f['min_occupancy'] for f in real_features])
    std_occupancy = np.mean([f['std_occupancy'] for f in real_features])
    total_capacity = np.mean([f['total_capacity'] for f in real_features])
    avg_capacity = np.mean([f['avg_capacity'] for f in real_features])
    max_capacity = np.mean([f['max_capacity'] for f in real_features])
    min_capacity = np.mean([f['min_capacity'] for f in real_features])
    shelter_count = np.mean([f['shelter_count'] for f in real_features])
    org_count = np.mean([f['org_count'] for f in real_features])
    utilization_rate = np.mean([f['utilization_rate'] for f in real_features])
    
    features = [
        avg_occupancy,
        max_occupancy,
        min_occupancy,
        std_occupancy,
        total_capacity,
        avg_capacity,
        max_capacity,
        min_capacity,
        shelter_count,
        org_count,
        utilization_rate,
        day_of_week,
        month,
        year,
        is_weekend,
        is_winter,
        is_summer,
        is_spring,
        is_fall
    ]
    return features

@app.post("/predict")
async def predict(req: PredictRequest):
    if not model:
        return {"error": "Model not loaded", "beds_needed": 0, "meals_needed": 0, "kits_needed": 0, "confidence": 0.0}
    
    try:
        # Create features for current prediction
        features = create_current_features()
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Calculate confidence based on feature quality
        confidence = min(0.95, 0.7 + (len(real_features) / 1000) * 0.25)
        
        # Convert total occupancy to individual needs
        beds_needed = max(0, int(prediction * 0.8))  # 80% of total need beds
        meals_needed = max(0, int(prediction * 1.2))  # 120% of total need meals
        kits_needed = max(0, int(prediction * 0.3))   # 30% of total need kits
        
        return {
            "beds_needed": beds_needed,
            "meals_needed": meals_needed,
            "kits_needed": kits_needed,
            "total_occupancy": int(prediction),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "data_source": "Real Toronto Shelter Data (2017-2020)",
            "model_info": {
                "features_used": len(features),
                "training_samples": len(real_features) if real_features else 0,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "prediction_time": datetime.now().strftime("%H:%M:%S")
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "beds_needed": 0, "meals_needed": 0, "kits_needed": 0, "confidence": 0.0}

@app.get("/")
async def root():
    return {"status": "ok", "message": "ML microservice running", "model_loaded": model is not None} 