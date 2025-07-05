#!/usr/bin/env python3
"""
Use Trained Model Script
Loads the trained best_model.h5 and makes predictions with location features
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import holidays
import warnings
warnings.filterwarnings('ignore')

# GPU Configuration
print("Configuring GPU for TensorFlow...")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled successfully!")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")

# Set mixed precision for better performance
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled for better performance!")
except:
    print("Mixed precision not available, using default precision.")

# ============================================================================
# LOCATION FEATURES PIPELINE (Same as in notebook)
# ============================================================================

class TorontoSectorMapper:
    """Toronto Sector Mapper for Shelter Location Features"""
    
    def __init__(self):
        # Define Toronto sectors based on postal code areas
        self.toronto_sectors = {
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
        self.sector_indicators = {
            'downtown_core': {'avg_income': 85000, 'population_density': 8500, 'transit_accessibility': 0.95, 'crime_rate': 0.3, 'homelessness_rate': 0.8},
            'east_end': {'avg_income': 65000, 'population_density': 4200, 'transit_accessibility': 0.75, 'crime_rate': 0.4, 'homelessness_rate': 0.6},
            'west_end': {'avg_income': 72000, 'population_density': 5800, 'transit_accessibility': 0.85, 'crime_rate': 0.35, 'homelessness_rate': 0.7},
            'north_end': {'avg_income': 95000, 'population_density': 3200, 'transit_accessibility': 0.65, 'crime_rate': 0.2, 'homelessness_rate': 0.3},
            'etobicoke': {'avg_income': 78000, 'population_density': 2800, 'transit_accessibility': 0.55, 'crime_rate': 0.25, 'homelessness_rate': 0.4},
            'york': {'avg_income': 68000, 'population_density': 3800, 'transit_accessibility': 0.70, 'crime_rate': 0.45, 'homelessness_rate': 0.5}
        }
        
        # Initialize sector encoder with all known sectors plus 'unknown'
        all_sectors = list(self.toronto_sectors.keys()) + ['unknown']
        self.sector_encoder = LabelEncoder()
        self.sector_encoder.fit(all_sectors)
        
    def get_sector_from_postal_code(self, postal_code):
        if pd.isna(postal_code) or postal_code == '':
            return 'unknown'
        fsa = str(postal_code).strip()[:3].upper()
        for sector_id, sector_info in self.toronto_sectors.items():
            if fsa in sector_info['postal_codes']:
                return sector_id
        return 'unknown'
    
    def get_sector_for_new_shelter(self, address, postal_code=None, city='Toronto', province='ON'):
        if postal_code:
            sector_id = self.get_sector_from_postal_code(postal_code)
        else:
            sector_id = 'unknown'
        
        # Ensure sector_id is in the encoder classes
        if sector_id not in self.sector_encoder.classes_:
            print(f"Warning: Unknown sector '{sector_id}', mapping to 'unknown'")
            sector_id = 'unknown'
        
        sector_info = {
            'sector_id': sector_id,
            'sector_name': self.toronto_sectors.get(sector_id, {}).get('name', 'Unknown'),
            'sector_description': self.toronto_sectors.get(sector_id, {}).get('description', 'Unknown'),
            'socioeconomic_indicators': self.sector_indicators.get(sector_id, {}),
            'sector_encoded': self.sector_encoder.transform([sector_id])[0]
        }
        
        return sector_info

class LocationFeaturePipeline:
    def __init__(self):
        self.sector_mapper = TorontoSectorMapper()
        
    def get_location_features_for_prediction(self, shelter_info):
        sector_info = self.sector_mapper.get_sector_for_new_shelter(
            address=shelter_info.get('address', ''),
            postal_code=shelter_info.get('postal_code'),
            city=shelter_info.get('city', 'Toronto'),
            province=shelter_info.get('province', 'ON')
        )
        
        features = {
            'sector_id': sector_info['sector_id'],
            'sector_encoded': sector_info['sector_encoded'],
            'sector_avg_income': sector_info['socioeconomic_indicators'].get('avg_income', 0),
            'sector_population_density': sector_info['socioeconomic_indicators'].get('population_density', 0),
            'sector_transit_accessibility': sector_info['socioeconomic_indicators'].get('transit_accessibility', 0),
            'sector_crime_rate': sector_info['socioeconomic_indicators'].get('crime_rate', 0),
            'sector_homelessness_rate': sector_info['socioeconomic_indicators'].get('homelessness_rate', 0)
        }
        
        return features

# ============================================================================
# PREDICTION CLASS
# ============================================================================

class TrainedModelPredictor:
    def __init__(self, model_path='best_model.h5', scaler_path=None):
        """Initialize predictor with trained model"""
        self.model = None
        self.scaler = None
        self.location_pipeline = LocationFeaturePipeline()
        self.canada_holidays = holidays.Canada()
        self.sequence_length = 7
        
        # Load model
        try:
            self.model = load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return
        
        # Load scaler if available
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded from {scaler_path}")
            except Exception as e:
                print(f"✗ Error loading scaler: {e}")
    
    def prepare_prediction_features(self, target_date, shelter_info):
        """Prepare features for prediction"""
        target_dt = pd.to_datetime(target_date)
        start_date = target_dt - timedelta(days=self.sequence_length)
        
        # Get location features
        location_features = self.location_pipeline.get_location_features_for_prediction(shelter_info)
        
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
    
    def predict_occupancy(self, shelter_info, target_date):
        """Predict occupancy for a specific shelter on a specific date"""
        print(f"Predicting occupancy for shelter: {shelter_info.get('name', 'Unknown')}")
        print(f"Target date: {target_date}")
        
        # Prepare features
        features = self.prepare_prediction_features(target_date, shelter_info)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_reshaped = features.reshape(-1, features.shape[-1])
            features_scaled = self.scaler.transform(features_reshaped)
            features = features_scaled.reshape(features.shape)
        
        # Make prediction
        prediction = self.model.predict(features.reshape(1, *features.shape))
        predicted_occupancy = prediction[0][0]
        
        # Apply shelter-specific scaling if capacity is provided
        if 'maxCapacity' in shelter_info:
            max_capacity = shelter_info['maxCapacity']
            scaled_prediction = min(predicted_occupancy * (max_capacity / 100), max_capacity)
            predicted_occupancy = max(0, scaled_prediction)
        
        # Get location info
        location_features = self.location_pipeline.get_location_features_for_prediction(shelter_info)
        
        return {
            'shelter_name': shelter_info.get('name', 'Unknown'),
            'target_date': target_date,
            'predicted_occupancy': round(predicted_occupancy, 0),
            'max_capacity': shelter_info.get('maxCapacity', None),
            'utilization_rate': round((predicted_occupancy / shelter_info.get('maxCapacity', 100)) * 100, 1) if shelter_info.get('maxCapacity') else None,
            'sector_info': {
                'sector_id': location_features.get('sector_id', 'unknown'),
                'sector_name': location_features.get('sector_name', 'Unknown'),
                'sector_description': location_features.get('sector_description', 'Unknown')
            },
            'location_features': location_features
        }

# ============================================================================
# MAIN USAGE
# ============================================================================

def main():
    """Example usage of the trained model"""
    print("Shelter Occupancy Prediction with Trained Model")
    print("=" * 50)
    
    # Initialize predictor
    predictor = TrainedModelPredictor(model_path='best_model.h5')
    
    if predictor.model is None:
        print("❌ Could not load model. Please ensure best_model.h5 exists.")
        return
    
    # Example predictions
    test_shelters = [
        {
            'name': 'Downtown Shelter',
            'maxCapacity': 100,
            'address': '100 Bay Street',
            'postal_code': 'M5J 2T3',
            'city': 'Toronto',
            'province': 'ON'
        },
        {
            'name': 'East End Shelter',
            'maxCapacity': 150,
            'address': '200 Danforth Avenue',
            'postal_code': 'M4K 1L6',
            'city': 'Toronto',
            'province': 'ON'
        },
        {
            'name': 'West End Shelter',
            'maxCapacity': 80,
            'address': '300 Queen Street West',
            'postal_code': 'M5V 2Z4',
            'city': 'Toronto',
            'province': 'ON'
        }
    ]
    
    # Predict for tomorrow
    tomorrow = datetime.now() + timedelta(days=1)
    target_date = tomorrow.strftime('%Y-%m-%d')
    
    print(f"\nPredictions for {target_date}:")
    print("-" * 50)
    
    for shelter in test_shelters:
        prediction = predictor.predict_occupancy(shelter, target_date)
        
        print(f"\nShelter: {prediction['shelter_name']}")
        print(f"Sector: {prediction['sector_info']['sector_name']}")
        print(f"Sector Description: {prediction['sector_info']['sector_description']}")
        print(f"Predicted Occupancy: {prediction['predicted_occupancy']}")
        print(f"Max Capacity: {prediction['max_capacity']}")
        print(f"Utilization Rate: {prediction['utilization_rate']}%")
        print("-" * 30)

if __name__ == "__main__":
    main() 