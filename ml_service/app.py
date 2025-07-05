from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_service.model import ShelterMLService

app = FastAPI(
    title="Shelter Prediction API",
    description="ML service for predicting shelter occupancy and resource needs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML service
ml_service = ShelterMLService()

class PredictionResponse(BaseModel):
    shelter_id: str
    shelter_name: str
    current_occupancy: int
    capacity: int
    predictions: List[dict]

class OverviewResponse(BaseModel):
    total_shelters: int
    prediction_days: int
    predictions: List[dict]

@app.on_event("startup")
async def startup_event():
    """Initialize the ML service on startup"""
    print("Starting ML service...")
    try:
        # Load or train the model
        if not ml_service.load_model():
            print("Training new model...")
            ml_service.train_model(epochs=50)
        print("ML service ready!")
    except Exception as e:
        print(f"Error initializing ML service: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "shelter-prediction-api",
        "model_loaded": ml_service.model is not None
    }

@app.get("/predict/shelter/{shelter_id}")
async def predict_shelter(shelter_id: str, days: int = 7):
    """Predict occupancy for a specific shelter"""
    try:
        prediction = ml_service.predict_shelter(shelter_id, days)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Shelter not found")
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/overview")
async def predict_overview(days: int = 7):
    """Predict overview for all shelters"""
    try:
        overview = ml_service.predict_overview(days)
        if overview is None:
            raise HTTPException(status_code=500, detail="Failed to generate overview")
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/resources/{shelter_id}")
async def predict_resources(shelter_id: str, days: int = 7):
    """Predict resource needs for a shelter"""
    try:
        # Get shelter prediction
        prediction = ml_service.predict_shelter(shelter_id, days)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Shelter not found")
        
        # Calculate resource predictions based on occupancy
        resource_predictions = []
        for pred in prediction['predictions']:
            occupancy = pred['predicted_occupancy']
            
            # Simple resource calculations (can be enhanced)
            meals_needed = max(occupancy * 2.5, 0)  # 2.5 meals per person
            kits_needed = max(occupancy * 0.8, 0)   # 0.8 kits per person
            staff_needed = max(occupancy / 20, 1)    # 1 staff per 20 people
            
            resource_predictions.append({
                'date': pred['date'],
                'predicted_occupancy': occupancy,
                'meals_needed': round(meals_needed),
                'kits_needed': round(kits_needed),
                'staff_needed': round(staff_needed, 1),
                'utilization_rate': pred['utilization_rate']
            })
        
        return {
            'shelter_id': shelter_id,
            'shelter_name': prediction['shelter_name'],
            'capacity': prediction['capacity'],
            'resource_predictions': resource_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with current data"""
    try:
        success = ml_service.train_model(epochs=100)
        if success:
            return {"message": "Model retrained successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to retrain model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def model_status():
    """Get model status and information"""
    return {
        "model_loaded": ml_service.model is not None,
        "device": str(ml_service.device),
        "data_loaded": ml_service.shelter_data is not None,
        "total_shelters": len(ml_service.shelter_data) if ml_service.shelter_data else 0,
        "total_features": len(ml_service.features_data) if ml_service.features_data else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 