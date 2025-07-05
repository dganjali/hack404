import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any

class DemandForecaster:
    def __init__(self):
        self.beds_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.meals_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.kits_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.toronto_features = None
        
    def load_toronto_features(self):
        """Load Toronto features for enhanced prediction"""
        try:
            with open("toronto_features.json", "r") as f:
                self.toronto_features = json.load(f)
            print(f"Loaded {len(self.toronto_features)} Toronto feature records")
        except FileNotFoundError:
            print("Toronto features not found, using basic prediction")
            self.toronto_features = None
        
    def _prepare_features(self, history: List[Dict]) -> tuple:
        """Prepare features for the ML model with Toronto data enhancement"""
        if len(history) < 7:
            raise ValueError("Need at least 7 days of history")
        
        # Sort by date
        history.sort(key=lambda x: x['date'])
        
        # Create features
        features = []
        beds_targets = []
        meals_targets = []
        kits_targets = []
        
        for i in range(7, len(history)):
            # Past 7 days of data
            past_week = history[i-7:i]
            
            # Basic features: rolling statistics
            beds_values = [h['beds_needed'] for h in past_week]
            meals_values = [h['meals_needed'] for h in past_week]
            kits_values = [h['kits_needed'] for h in past_week]
            
            # Create basic feature vector
            feature_vector = [
                np.mean(beds_values), np.std(beds_values), np.max(beds_values), np.min(beds_values),
                np.mean(meals_values), np.std(meals_values), np.max(meals_values), np.min(meals_values),
                np.mean(kits_values), np.std(kits_values), np.max(kits_values), np.min(kits_values),
                # Day of week (0-6)
                datetime.strptime(history[i]['date'], '%Y-%m-%d').weekday(),
                # Trend (difference from 7 days ago)
                beds_values[-1] - beds_values[0],
                meals_values[-1] - meals_values[0],
                kits_values[-1] - kits_values[0]
            ]
            
            # Add Toronto-specific features if available
            if self.toronto_features:
                # Find corresponding Toronto data for this date
                toronto_data = None
                for tf in self.toronto_features:
                    if tf['date'] == history[i]['date']:
                        toronto_data = tf
                        break
                
                if toronto_data:
                    # Add demographic features
                    feature_vector.extend([
                        toronto_data.get('total_actively_homeless', 0),
                        toronto_data.get('total_newly_identified', 0),
                        toronto_data.get('total_moved_to_housing', 0),
                        toronto_data.get('net_flow', 0),
                        toronto_data.get('turnover_rate', 0),
                        toronto_data.get('housing_success_rate', 0),
                        
                        # Demographic breakdown
                        toronto_data.get('total_under16', 0),
                        toronto_data.get('total_16_24', 0),
                        toronto_data.get('total_25_34', 0),
                        toronto_data.get('total_35_44', 0),
                        toronto_data.get('total_45_54', 0),
                        toronto_data.get('total_55_64', 0),
                        toronto_data.get('total_65_over', 0),
                        
                        # Gender breakdown
                        toronto_data.get('total_male', 0),
                        toronto_data.get('total_female', 0),
                        toronto_data.get('total_transgender', 0),
                        
                        # Population group percentages
                        toronto_data.get('chronic_percentage', 0),
                        toronto_data.get('refugee_percentage', 0),
                        toronto_data.get('family_percentage', 0),
                        toronto_data.get('youth_percentage', 0),
                        toronto_data.get('single_adult_percentage', 0)
                    ])
                else:
                    # Add zeros for missing Toronto data
                    feature_vector.extend([0] * 25)
            else:
                # Add zeros for missing Toronto data
                feature_vector.extend([0] * 25)
            
            features.append(feature_vector)
            beds_targets.append(history[i]['beds_needed'])
            meals_targets.append(history[i]['meals_needed'])
            kits_targets.append(history[i]['kits_needed'])
        
        return np.array(features), np.array(beds_targets), np.array(meals_targets), np.array(kits_targets)
    
    def train(self, history: List[Dict]):
        """Train the forecasting models with Toronto data"""
        if len(history) < 14:
            raise ValueError("Need at least 14 days of history for training")
        
        # Load Toronto features
        self.load_toronto_features()
        
        features, beds_targets, meals_targets, kits_targets = self._prepare_features(history)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train models
        self.beds_model.fit(features_scaled, beds_targets)
        self.meals_model.fit(features_scaled, meals_targets)
        self.kits_model.fit(features_scaled, kits_targets)
        
        print(f"✅ Trained models with {len(features)} samples and {features.shape[1]} features")
    
    def predict(self, history: List[Dict]) -> Dict[str, int]:
        """Predict tomorrow's demand with Toronto data enhancement"""
        if len(history) < 7:
            # If insufficient data, return simple average
            beds_avg = int(np.mean([h['beds_needed'] for h in history]))
            meals_avg = int(np.mean([h['meals_needed'] for h in history]))
            kits_avg = int(np.mean([h['kits_needed'] for h in history]))
            return {
                "beds": max(0, beds_avg),
                "meals": max(0, meals_avg),
                "kits": max(0, kits_avg)
            }
        
        # Load Toronto features if not already loaded
        if self.toronto_features is None:
            self.load_toronto_features()
        
        # Prepare features for prediction
        history.sort(key=lambda x: x['date'])
        past_week = history[-7:]
        
        beds_values = [h['beds_needed'] for h in past_week]
        meals_values = [h['meals_needed'] for h in past_week]
        kits_values = [h['kits_needed'] for h in past_week]
        
        # Create feature vector for tomorrow
        tomorrow = datetime.strptime(history[-1]['date'], '%Y-%m-%d') + timedelta(days=1)
        feature_vector = [
            np.mean(beds_values), np.std(beds_values), np.max(beds_values), np.min(beds_values),
            np.mean(meals_values), np.std(meals_values), np.max(meals_values), np.min(meals_values),
            np.mean(kits_values), np.std(kits_values), np.max(kits_values), np.min(kits_values),
            tomorrow.weekday(),
            beds_values[-1] - beds_values[0],
            meals_values[-1] - meals_values[0],
            kits_values[-1] - kits_values[0]
        ]
        
        # Add Toronto-specific features for tomorrow
        if self.toronto_features:
            # Find the most recent Toronto data
            latest_toronto = max(self.toronto_features, key=lambda x: x['date'])
            
            # Add demographic features
            feature_vector.extend([
                latest_toronto.get('total_actively_homeless', 0),
                latest_toronto.get('total_newly_identified', 0),
                latest_toronto.get('total_moved_to_housing', 0),
                latest_toronto.get('net_flow', 0),
                latest_toronto.get('turnover_rate', 0),
                latest_toronto.get('housing_success_rate', 0),
                
                # Demographic breakdown
                latest_toronto.get('total_under16', 0),
                latest_toronto.get('total_16_24', 0),
                latest_toronto.get('total_25_34', 0),
                latest_toronto.get('total_35_44', 0),
                latest_toronto.get('total_45_54', 0),
                latest_toronto.get('total_55_64', 0),
                latest_toronto.get('total_65_over', 0),
                
                # Gender breakdown
                latest_toronto.get('total_male', 0),
                latest_toronto.get('total_female', 0),
                latest_toronto.get('total_transgender', 0),
                
                # Population group percentages
                latest_toronto.get('chronic_percentage', 0),
                latest_toronto.get('refugee_percentage', 0),
                latest_toronto.get('family_percentage', 0),
                latest_toronto.get('youth_percentage', 0),
                latest_toronto.get('single_adult_percentage', 0)
            ])
        else:
            # Add zeros for missing Toronto data
            feature_vector.extend([0] * 25)
        
        # Scale features
        features_scaled = self.scaler.transform([feature_vector])
        
        # Make predictions
        beds_pred = max(0, int(self.beds_model.predict(features_scaled)[0]))
        meals_pred = max(0, int(self.meals_model.predict(features_scaled)[0]))
        kits_pred = max(0, int(self.kits_model.predict(features_scaled)[0]))
        
        return {
            "beds": beds_pred,
            "meals": meals_pred,
            "kits": kits_pred
        }
    
    def predict_with_confidence(self, history: List[Dict]) -> Dict[str, Any]:
        """Predict with confidence intervals using Toronto data"""
        prediction = self.predict(history)
        
        # Calculate confidence based on historical variance and Toronto data
        if len(history) >= 7:
            beds_std = np.std([h['beds_needed'] for h in history[-7:]])
            meals_std = np.std([h['meals_needed'] for h in history[-7:]])
            kits_std = np.std([h['kits_needed'] for h in history[-7:]])
            
            # Adjust confidence based on Toronto data availability
            confidence_multiplier = 1.0
            if self.toronto_features:
                confidence_multiplier = 0.8  # More confident with Toronto data
            
            confidence = {
                "beds": {
                    "prediction": prediction["beds"],
                    "confidence_low": max(0, prediction["beds"] - int(beds_std * confidence_multiplier)),
                    "confidence_high": prediction["beds"] + int(beds_std * confidence_multiplier)
                },
                "meals": {
                    "prediction": prediction["meals"],
                    "confidence_low": max(0, prediction["meals"] - int(meals_std * confidence_multiplier)),
                    "confidence_high": prediction["meals"] + int(meals_std * confidence_multiplier)
                },
                "kits": {
                    "prediction": prediction["kits"],
                    "confidence_low": max(0, prediction["kits"] - int(kits_std * confidence_multiplier)),
                    "confidence_high": prediction["kits"] + int(kits_std * confidence_multiplier)
                }
            }
        else:
            confidence = {
                "beds": {"prediction": prediction["beds"], "confidence_low": prediction["beds"], "confidence_high": prediction["beds"]},
                "meals": {"prediction": prediction["meals"], "confidence_low": prediction["meals"], "confidence_high": prediction["meals"]},
                "kits": {"prediction": prediction["kits"], "confidence_low": prediction["kits"], "confidence_high": prediction["kits"]}
            }
        
        return confidence 