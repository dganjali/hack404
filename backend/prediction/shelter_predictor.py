#!/usr/bin/env python3
"""
Shelter Occupancy Predictor
A simplified interface for predicting shelter occupancy using trained models with location features.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from tensorflow import keras
import holidays

class ShelterPredictor:
    """Main class for making shelter occupancy predictions with location features"""
    
    def __init__(self, model_path='models/best_model.h5'):
        """Initialize the predictor with trained model and preprocessors"""
        self.model = None
        self.scaler = None
        self.canada_holidays = holidays.Canada()
        self.sequence_length = 7
        
        # Load model and preprocessors
        self._load_model_and_preprocessors(model_path)
    
    def _load_model_and_preprocessors(self, model_path):
        """Load the trained model and preprocessors"""
        try:
            # Load model
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
            
            # Load scaler if available
            scaler_path = 'models/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✓ Scaler loaded successfully")
            else:
                print("⚠ No scaler found, predictions will be unscaled")
            
        except Exception as e:
            print(f"✗ Error loading model/preprocessors: {e}")
            raise
    
    def predict_occupancy(self, shelter_info, target_date):
        """Predict occupancy for a specific shelter on a specific date"""
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        # Convert date to datetime if needed
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Create features for prediction
        features = self._create_prediction_features(target_date, shelter_info)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_reshaped = features.reshape(-1, features.shape[-1])
            features_scaled = self.scaler.transform(features_reshaped)
            features = features_scaled.reshape(features.shape)
        
        # Make prediction
        prediction = self.model.predict(features.reshape(1, *features.shape), verbose=0)
        predicted_occupancy = prediction[0][0]
        
        # Apply shelter-specific scaling if capacity is provided
        if 'maxCapacity' in shelter_info:
            max_capacity = shelter_info['maxCapacity']
            scaled_prediction = min(predicted_occupancy * (max_capacity / 100), max_capacity)
            predicted_occupancy = max(0, scaled_prediction)
        
        # Get location info
        location_features = self._get_location_features(shelter_info)
        
        return {
            'shelter_name': shelter_info.get('name', 'Unknown'),
            'target_date': target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date),
            'predicted_occupancy': round(predicted_occupancy, 0),
            'max_capacity': shelter_info.get('maxCapacity', None),
            'utilization_rate': round((predicted_occupancy / shelter_info.get('maxCapacity', 100)) * 100, 1) if shelter_info.get('maxCapacity') else None,
            'sector_info': {
                'sector_id': location_features['sector_id'],
                'sector_name': location_features['sector_name'],
                'sector_description': location_features['sector_description']
            },
            'location_features': location_features
        }
    
    def _create_prediction_features(self, target_date, shelter_info):
        """Create features for prediction with location features"""
        target_dt = pd.to_datetime(target_date)
        start_date = target_dt - timedelta(days=self.sequence_length)
        
        # Get location features
        location_features = self._get_location_features(shelter_info)
        
        sequence_features = []
        
        for i in range(self.sequence_length):
            current_date = start_date + timedelta(days=i)
            
            # Temporal features
            features = {
                'year': current_date.year,
                'month': current_date.month,
                'day_of_week': current_date.dayofweek,
                'day_of_month': current_date.day,
                'week_of_year': current_date.isocalendar().week,
                'quarter': current_date.quarter,
                'season': self._get_season(current_date.month),
                'is_weekend': int(current_date.dayofweek >= 5),
                'is_month_end': int(current_date.is_month_end),
                'is_month_start': int(current_date.is_month_start),
                'is_holiday': int(current_date in self.canada_holidays),
                'temperature': self._simulate_temperature(current_date),
                'precipitation_mm': self._simulate_precipitation(current_date),
                'weather_severity': self._get_weather_severity(current_date)
            }
            
            # Add location features
            features.update({
                'sector_encoded': location_features['sector_encoded'],
                'sector_avg_income': location_features['sector_avg_income'] / 100000,  # Normalize
                'sector_population_density': location_features['sector_population_density'] / 10000,  # Normalize
                'sector_transit_accessibility': location_features['sector_transit_accessibility'],
                'sector_crime_rate': location_features['sector_crime_rate'],
                'sector_homelessness_rate': location_features['sector_homelessness_rate']
            })
            
            sequence_features.append(list(features.values()))
        
        return np.array(sequence_features)
    
    def _get_location_features(self, shelter_info):
        """Get location features for a shelter"""
        # Extract postal code from address or use provided postal code
        postal_code = shelter_info.get('postal_code')
        if not postal_code and 'address' in shelter_info:
            # Try to extract postal code from address
            address = shelter_info['address']
            # Simple extraction - look for pattern like "M5A 1A1" or "M5A"
            import re
            postal_match = re.search(r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b', address.upper())
            if postal_match:
                postal_code = postal_match.group()
        
        # Get sector information
        sector_info = self._get_sector_for_shelter(postal_code)
        
        return {
            'sector_id': sector_info['sector_id'],
            'sector_name': sector_info['sector_name'],
            'sector_description': sector_info['sector_description'],
            'sector_encoded': sector_info['sector_encoded'],
            'sector_avg_income': sector_info['socioeconomic_indicators'].get('avg_income', 0),
            'sector_population_density': sector_info['socioeconomic_indicators'].get('population_density', 0),
            'sector_transit_accessibility': sector_info['socioeconomic_indicators'].get('transit_accessibility', 0),
            'sector_crime_rate': sector_info['socioeconomic_indicators'].get('crime_rate', 0),
            'sector_homelessness_rate': sector_info['socioeconomic_indicators'].get('homelessness_rate', 0)
        }
    
    def _get_sector_for_shelter(self, postal_code):
        """Get sector information for a shelter based on postal code"""
        # Define Toronto sectors based on postal code areas
        toronto_sectors = {
            'downtown_core': {
                'name': 'Downtown Core',
                'postal_codes': ['M5A', 'M5B', 'M5C', 'M5E', 'M5G', 'M5H', 'M5J', 'M5K', 'M5L', 'M5M', 'M5N', 'M5P', 'M5R', 'M5S', 'M5T', 'M5V', 'M5W', 'M5X', 'M5Y', 'M5Z'],
                'description': 'Financial district, entertainment district, university area'
            },
            'east_end': {
                'name': 'East End',
                'postal_codes': ['M1B', 'M1C', 'M1E', 'M1G', 'M1H', 'M1J', 'M1K', 'M1L', 'M1M', 'M1N', 'M1P', 'M1R', 'M1S', 'M1T', 'M1V', 'M1W', 'M1X'],
                'description': 'Scarborough, East York, Beaches'
            },
            'west_end': {
                'name': 'West End',
                'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S'],
                'description': 'West Toronto, Parkdale, High Park, Junction'
            },
            'north_end': {
                'name': 'North End',
                'postal_codes': ['M2H', 'M2J', 'M2K', 'M2L', 'M2M', 'M2N', 'M2P', 'M2R', 'M3A', 'M3B', 'M3C', 'M3H', 'M3J', 'M3K', 'M3L', 'M3M', 'M3N', 'M4A', 'M4B', 'M4C', 'M4E', 'M4G', 'M4H', 'M4J', 'M4K', 'M4L', 'M4M', 'M4N', 'M4P', 'M4R', 'M4S', 'M4T', 'M4V', 'M4W', 'M4X', 'M4Y'],
                'description': 'North York, York, Don Mills, Lawrence Park'
            },
            'etobicoke': {
                'name': 'Etobicoke',
                'postal_codes': ['M8V', 'M8W', 'M8X', 'M8Y', 'M8Z', 'M9A', 'M9B', 'M9C', 'M9P', 'M9R', 'M9V', 'M9W'],
                'description': 'Etobicoke, Rexdale, Humber Bay'
            },
            'york': {
                'name': 'York',
                'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S'],
                'description': 'York, Weston, Mount Dennis'
            }
        }
        
        # Sector socioeconomic indicators
        sector_indicators = {
            'downtown_core': {'avg_income': 85000, 'population_density': 8500, 'transit_accessibility': 0.95, 'crime_rate': 0.3, 'homelessness_rate': 0.8},
            'east_end': {'avg_income': 65000, 'population_density': 4200, 'transit_accessibility': 0.75, 'crime_rate': 0.4, 'homelessness_rate': 0.6},
            'west_end': {'avg_income': 72000, 'population_density': 5800, 'transit_accessibility': 0.85, 'crime_rate': 0.35, 'homelessness_rate': 0.7},
            'north_end': {'avg_income': 95000, 'population_density': 3200, 'transit_accessibility': 0.65, 'crime_rate': 0.2, 'homelessness_rate': 0.3},
            'etobicoke': {'avg_income': 78000, 'population_density': 2800, 'transit_accessibility': 0.55, 'crime_rate': 0.25, 'homelessness_rate': 0.4},
            'york': {'avg_income': 68000, 'population_density': 3800, 'transit_accessibility': 0.70, 'crime_rate': 0.45, 'homelessness_rate': 0.5}
        }
        
        # Determine sector from postal code
        sector_id = 'unknown'
        if postal_code:
            fsa = str(postal_code).strip()[:3].upper()
            for sector_key, sector_info in toronto_sectors.items():
                if fsa in sector_info['postal_codes']:
                    sector_id = sector_key
                    break
        
        # Get sector info
        sector_info = toronto_sectors.get(sector_id, {
            'name': 'Unknown',
            'description': 'Unknown area'
        })
        
        # Create sector encoder mapping
        sector_encoder = {
            'downtown_core': 0,
            'east_end': 1,
            'west_end': 2,
            'north_end': 3,
            'etobicoke': 4,
            'york': 5,
            'unknown': 6
        }
        
        return {
            'sector_id': sector_id,
            'sector_name': sector_info['name'],
            'sector_description': sector_info['description'],
            'sector_encoded': sector_encoder.get(sector_id, 6),
            'socioeconomic_indicators': sector_indicators.get(sector_id, {})
        }
    
    def _get_season(self, month):
        if month in [12, 1, 2]: return 1  # Winter
        elif month in [3, 4, 5]: return 2  # Spring
        elif month in [6, 7, 8]: return 3  # Summer
        else: return 4  # Fall
    
    def _simulate_temperature(self, date):
        season = self._get_season(date.month)
        if season == 1: return np.random.normal(-5, 10)
        elif season == 2: return np.random.normal(10, 8)
        elif season == 3: return np.random.normal(25, 8)
        else: return np.random.normal(15, 8)
    
    def _simulate_precipitation(self, date):
        season = self._get_season(date.month)
        precip_prob = {1: 0.3, 2: 0.4, 3: 0.2, 4: 0.3}[season]
        if np.random.random() < precip_prob:
            return np.random.exponential(5)
        return 0
    
    def _get_weather_severity(self, date):
        temp = self._simulate_temperature(date)
        precip = self._simulate_precipitation(date)
        if (temp < -10) or (temp > 35) or (precip > 20): return 3
        elif (temp < 0) or (temp > 25) or (precip > 10): return 2
        else: return 1
    
    def predict_date_range(self, shelter_info, start_date, end_date):
        """Predict occupancy for a range of dates"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        predictions = []
        
        for date in date_range:
            prediction = self.predict_occupancy(shelter_info, date)
            predictions.append(prediction)
        
        return predictions

def main():
    """Example usage of the ShelterPredictor"""
    try:
        # Initialize predictor
        predictor = ShelterPredictor()
        
        # Example shelter info
        shelter_info = {
            'name': 'Downtown Shelter',
            'maxCapacity': 100,
            'address': '100 Bay Street, Toronto, ON M5J 2T3',
            'postal_code': 'M5J 2T3'
        }
        
        # Example prediction
        date = "2024-01-15"
        prediction = predictor.predict_occupancy(shelter_info, date)
        print(f"Prediction for {shelter_info['name']} on {date}:")
        print(f"Predicted occupancy: {prediction['predicted_occupancy']}")
        print(f"Sector: {prediction['sector_info']['sector_name']}")
        print(f"Utilization rate: {prediction['utilization_rate']}%")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 