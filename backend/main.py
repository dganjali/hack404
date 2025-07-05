from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from dotenv import load_dotenv
from predict import DemandForecaster
from optimize import ResourceOptimizer
from database import connect_to_mongo, close_mongo_connection, init_database
from auth import login_user, register_user, get_current_user, get_user_from_token

# Load environment variables
load_dotenv()

app = FastAPI(title="NeedsMatcher API", version="1.0.0")

# Security
security = HTTPBearer()

# Get CORS origins from environment
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Event handlers for MongoDB
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    await init_database()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# Load data with Toronto data priority
def load_shelters():
    """Load shelter data, preferring Toronto data if available"""
    try:
        # Try Toronto data first
        with open("toronto_shelters.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            # Fall back to mock data
            with open("shelters.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

def load_intake_history():
    """Load intake history, preferring Toronto data if available"""
    try:
        # Try Toronto data first
        with open("toronto_intake_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            # Fall back to mock data
            with open("intake_history.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

def load_toronto_features():
    """Load Toronto features for enhanced ML"""
    try:
        with open("toronto_features.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Pydantic models
class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str
    name: str = ""

class Shelter(BaseModel):
    id: str
    name: str
    capacity: int
    current_beds: int
    current_meals: int
    current_kits: int

class ForecastRequest(BaseModel):
    shelter_id: str

class ForecastResponse(BaseModel):
    shelter_id: str
    predicted_beds_needed: int
    predicted_meals_needed: int
    predicted_kits_needed: int

class TransferPlan(BaseModel):
    from_shelter: str
    to_shelter: str
    item: str
    amount: int

class OptimizationResponse(BaseModel):
    transfers: List[TransferPlan]
    shortages_reduced: float

class TorontoFeature(BaseModel):
    date: str
    total_actively_homeless: int
    total_newly_identified: int
    total_moved_to_housing: int
    net_flow: int
    turnover_rate: float
    housing_success_rate: float

# Authentication dependency
async def get_current_user_dependency(credentials: HTTPAuthorizationCredentials = Depends(security)):
    user = get_user_from_token(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@app.get("/")
async def root():
    return {"message": "NeedsMatcher API is running!", "data_source": "Toronto Shelter System"}

# Authentication endpoints
@app.post("/auth/register")
async def register(user_data: UserRegister):
    """Register a new user"""
    result = await register_user(user_data.email, user_data.password, user_data.name)
    if result["success"]:
        return {"message": result["message"]}
    else:
        raise HTTPException(status_code=400, detail=result["message"])

@app.post("/auth/login")
async def login(user_data: UserLogin):
    """Login user and get access token"""
    result = await login_user(user_data.email, user_data.password)
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=401, detail=result["message"])

@app.get("/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user_dependency)):
    """Get current user information"""
    return current_user

# Protected endpoints
@app.get("/shelters")
async def get_shelters(current_user: dict = Depends(get_current_user_dependency)):
    """Get all shelters with current inventory"""
    shelters = load_shelters()
    return {"shelters": shelters}

@app.get("/intake-history")
async def get_intake_history(current_user: dict = Depends(get_current_user_dependency)):
    """Get historical intake data"""
    history = load_intake_history()
    return {"history": history}

@app.get("/toronto-features")
async def get_toronto_features(current_user: dict = Depends(get_current_user_dependency)):
    """Get Toronto demographic and flow features"""
    features = load_toronto_features()
    if features:
        return {"features": features}
    else:
        raise HTTPException(status_code=404, detail="Toronto features not available")

@app.get("/data-summary")
async def get_data_summary(current_user: dict = Depends(get_current_user_dependency)):
    """Get summary of current data sources and statistics"""
    shelters = load_shelters()
    history = load_intake_history()
    features = load_toronto_features()
    
    summary = {
        "data_source": "Toronto Shelter System" if features else "Mock Data",
        "shelters_count": len(shelters),
        "history_records": len(history),
        "features_available": features is not None,
        "date_range": None
    }
    
    if history:
        dates = [h['date'] for h in history]
        summary["date_range"] = {
            "start": min(dates),
            "end": max(dates)
        }
    
    if features:
        summary["toronto_features"] = {
            "total_records": len(features),
            "latest_actively_homeless": features[-1]['total_actively_homeless'],
            "latest_housing_success_rate": features[-1]['housing_success_rate']
        }
    
    return summary

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_demand(request: ForecastRequest, current_user: dict = Depends(get_current_user_dependency)):
    """Forecast tomorrow's demand for a specific shelter"""
    try:
        forecaster = DemandForecaster()
        shelters = load_shelters()
        history = load_intake_history()
        
        # Find the shelter
        shelter = next((s for s in shelters if s["id"] == request.shelter_id), None)
        if not shelter:
            raise HTTPException(status_code=404, detail="Shelter not found")
        
        # Get historical data for this shelter
        shelter_history = [h for h in history if h["shelter_id"] == request.shelter_id]
        
        if len(shelter_history) < 7:
            raise HTTPException(status_code=400, detail="Insufficient historical data")
        
        # Make prediction
        prediction = forecaster.predict(shelter_history)
        
        return ForecastResponse(
            shelter_id=request.shelter_id,
            predicted_beds_needed=prediction["beds"],
            predicted_meals_needed=prediction["meals"],
            predicted_kits_needed=prediction["kits"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast-all")
async def forecast_all_shelters(current_user: dict = Depends(get_current_user_dependency)):
    """Forecast demand for all shelters"""
    try:
        forecaster = DemandForecaster()
        shelters = load_shelters()
        history = load_intake_history()
        
        predictions = []
        for shelter in shelters:
            shelter_history = [h for h in history if h["shelter_id"] == shelter["id"]]
            if len(shelter_history) >= 7:
                prediction = forecaster.predict(shelter_history)
                predictions.append({
                    "shelter_id": shelter["id"],
                    "shelter_name": shelter["name"],
                    "predicted_beds_needed": prediction["beds"],
                    "predicted_meals_needed": prediction["meals"],
                    "predicted_kits_needed": prediction["kits"]
                })
        
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_resources(current_user: dict = Depends(get_current_user_dependency)):
    """Optimize resource allocation between shelters"""
    try:
        optimizer = ResourceOptimizer()
        shelters = load_shelters()
        
        # Get forecasts for all shelters
        forecaster = DemandForecaster()
        history = load_intake_history()
        
        forecasts = {}
        for shelter in shelters:
            shelter_history = [h for h in history if h["shelter_id"] == shelter["id"]]
            if len(shelter_history) >= 7:
                prediction = forecaster.predict(shelter_history)
                forecasts[shelter["id"]] = prediction
        
        # Run optimization
        transfers, reduction = optimizer.optimize(shelters, forecasts)
        
        transfer_plans = [
            TransferPlan(
                from_shelter=t["from"],
                to_shelter=t["to"],
                item=t["item"],
                amount=t["amount"]
            ) for t in transfers
        ]
        
        return OptimizationResponse(
            transfers=transfer_plans,
            shortages_reduced=reduction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard-data")
async def get_dashboard_data(current_user: dict = Depends(get_current_user_dependency)):
    """Get all data needed for the dashboard"""
    try:
        shelters = load_shelters()
        history = load_intake_history()
        features = load_toronto_features()
        
        # Get forecasts
        forecaster = DemandForecaster()
        forecasts = {}
        for shelter in shelters:
            shelter_history = [h for h in history if h["shelter_id"] == shelter["id"]]
            if len(shelter_history) >= 7:
                prediction = forecaster.predict(shelter_history)
                forecasts[shelter["id"]] = prediction
        
        # Get optimization
        optimizer = ResourceOptimizer()
        transfers, reduction = optimizer.optimize(shelters, forecasts)
        
        response = {
            "shelters": shelters,
            "forecasts": forecasts,
            "transfers": transfers,
            "shortages_reduced": reduction,
            "history": history[-30:],  # Last 30 days
            "data_source": "Toronto Shelter System" if features else "Mock Data"
        }
        
        # Add Toronto features if available
        if features:
            response["toronto_features"] = features[-1]  # Latest features
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/toronto-analytics")
async def get_toronto_analytics(current_user: dict = Depends(get_current_user_dependency)):
    """Get Toronto-specific analytics and insights"""
    features = load_toronto_features()
    if not features:
        raise HTTPException(status_code=404, detail="Toronto features not available")
    
    # Calculate analytics
    latest = features[-1]
    previous = features[-2] if len(features) > 1 else latest
    
    analytics = {
        "current_stats": {
            "total_actively_homeless": latest['total_actively_homeless'],
            "total_newly_identified": latest['total_newly_identified'],
            "total_moved_to_housing": latest['total_moved_to_housing'],
            "housing_success_rate": latest['housing_success_rate'],
            "turnover_rate": latest['turnover_rate']
        },
        "trends": {
            "homeless_change": latest['total_actively_homeless'] - previous['total_actively_homeless'],
            "housing_success_change": latest['housing_success_rate'] - previous['housing_success_rate'],
            "net_flow": latest['net_flow']
        },
        "demographics": {
            "youth_percentage": latest['youth_percentage'],
            "family_percentage": latest['family_percentage'],
            "chronic_percentage": latest['chronic_percentage'],
            "refugee_percentage": latest['refugee_percentage']
        }
    }
    
    return analytics

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 