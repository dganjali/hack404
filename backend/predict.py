import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any
import os

class DemandForecaster:
    def __init__(self):
        self.model = None
        self.real_features = None
        self.load_trained_model()
        self.load_real_features()
        
    def load_trained_model(self):
        """Load the trained model from real data"""
        try:
            model_path = "trained_model.pkl"
            self.model = joblib.load(model_path)
            print(f"✅ Loaded trained model from {model_path}")
        except FileNotFoundError:
            print("❌ Trained model not found. Please run train_model.py first.")
            self.model = None
        
    def load_real_features(self):
        """Load real features for prediction"""
        try:
            features_path = "../data/real_features.json"
            with open(features_path, "r") as f:
                self.real_features = json.load(f)
            print(f"✅ Loaded {len(self.real_features)} real feature records")
        except FileNotFoundError:
            print("❌ Real features not found")
            self.real_features = None
    
    def _create_current_features(self) -> List[float]:
        """Create features for current date based on real data patterns, matching training order"""
        if not self.real_features:
            return self._create_basic_features()
        
        # Get current date
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
        avg_occupancy = np.mean([f['avg_occupancy'] for f in self.real_features])
        max_occupancy = np.mean([f['max_occupancy'] for f in self.real_features])
        min_occupancy = np.mean([f['min_occupancy'] for f in self.real_features])
        std_occupancy = np.mean([f['std_occupancy'] for f in self.real_features])
        total_capacity = np.mean([f['total_capacity'] for f in self.real_features])
        avg_capacity = np.mean([f['avg_capacity'] for f in self.real_features])
        max_capacity = np.mean([f['max_capacity'] for f in self.real_features])
        min_capacity = np.mean([f['min_capacity'] for f in self.real_features])
        shelter_count = np.mean([f['shelter_count'] for f in self.real_features])
        org_count = np.mean([f['org_count'] for f in self.real_features])
        utilization_rate = np.mean([f['utilization_rate'] for f in self.real_features])
        
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
    
    def _create_basic_features(self) -> List[float]:
        """Create basic features when real data is not available"""
        current_date = datetime.now()
        return [
            current_date.weekday(),
            current_date.month,
            current_date.day,
            100,  # Default occupancy
            20,   # Default std
            150,  # Default max
            50,   # Default min
            0,    # Default trend
            1.0,  # Default seasonal
            1.0,  # Default weekday
            100,  # Default capacity
            10,   # Default capacity std
            10,   # Default shelter count
            2,    # Default shelter std
            5,    # Default org count
            1     # Default org std
        ]
    
    def _get_recent_trend(self) -> float:
        """Get recent trend from real data"""
        if not self.real_features or len(self.real_features) < 7:
            return 0.0
        
        # Get last 7 days of data
        recent_data = sorted(self.real_features, key=lambda x: x['date'])[-7:]
        if len(recent_data) >= 2:
            return recent_data[-1]['total_occupancy'] - recent_data[0]['total_occupancy']
        return 0.0
    
    def _get_seasonal_factor(self, current_date: datetime) -> float:
        """Get seasonal factor based on month"""
        # Winter months (Dec-Feb) typically have higher demand
        if current_date.month in [12, 1, 2]:
            return 1.2
        # Summer months (Jun-Aug) typically have lower demand
        elif current_date.month in [6, 7, 8]:
            return 0.8
        else:
            return 1.0
    
    def _get_weekday_pattern(self, weekday: int) -> float:
        """Get weekday pattern factor"""
        # Weekends typically have different patterns
        if weekday in [5, 6]:  # Saturday, Sunday
            return 1.1
        else:
            return 1.0
    
    def predict_current_demand(self) -> Dict[str, Any]:
        """Predict current demand based on real-time data"""
        if not self.model:
            return {
                "error": "Model not loaded. Please run train_model.py first.",
                "beds_needed": 0,
                "meals_needed": 0,
                "kits_needed": 0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Create features for current prediction
        features = self._create_current_features()
        
        # Make prediction
        prediction = self.model.predict([features])[0]
        
        # Calculate confidence based on feature quality
        confidence = min(0.95, 0.7 + (len(self.real_features) / 1000) * 0.25)
        
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
                "training_samples": len(self.real_features) if self.real_features else 0,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "prediction_time": datetime.now().strftime("%H:%M:%S")
            }
        }
    
    def predict_future_demand(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Predict demand for the next N days"""
        if not self.model:
            return [{"error": "Model not loaded"}]
        
        predictions = []
        current_date = datetime.now()
        
        for i in range(1, days_ahead + 1):
            future_date = current_date + timedelta(days=i)
            
            # Create features for future date
            features = self._create_current_features()
            features[0] = future_date.weekday()  # Update day of week
            features[1] = future_date.month      # Update month
            features[2] = future_date.day        # Update day
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            
            # Convert to individual needs
            beds_needed = max(0, int(prediction * 0.8))
            meals_needed = max(0, int(prediction * 1.2))
            kits_needed = max(0, int(prediction * 0.3))
            
            predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "beds_needed": beds_needed,
                "meals_needed": meals_needed,
                "kits_needed": kits_needed,
                "total_occupancy": int(prediction),
                "day_of_week": future_date.strftime("%A"),
                "confidence": 0.85
            })
        
        return predictions
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get a summary of current predictions and data quality"""
        current_prediction = self.predict_current_demand()
        
        if "error" in current_prediction:
            return current_prediction
        
        # Get future predictions
        future_predictions = self.predict_future_demand(7)
        
        # Calculate statistics
        total_beds = sum(p.get('beds_needed', 0) for p in future_predictions)
        total_meals = sum(p.get('meals_needed', 0) for p in future_predictions)
        total_kits = sum(p.get('kits_needed', 0) for p in future_predictions)
        
        return {
            "current_prediction": current_prediction,
            "weekly_forecast": future_predictions,
            "summary": {
                "total_beds_weekly": total_beds,
                "total_meals_weekly": total_meals,
                "total_kits_weekly": total_kits,
                "avg_daily_beds": total_beds // 7,
                "avg_daily_meals": total_meals // 7,
                "avg_daily_kits": total_kits // 7,
                "data_quality": "Real Toronto Data (2017-2020)" if self.real_features else "Basic Features",
                "model_status": "Trained on Real Data" if self.model else "Not Trained"
            }
        } 