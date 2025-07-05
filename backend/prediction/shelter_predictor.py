#!/usr/bin/env python3
"""
Shelter Occupancy Predictor
A simplified interface for predicting shelter occupancy using trained models.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from tensorflow import keras

class ShelterPredictor:
    """Main class for making shelter occupancy predictions"""
    
    def __init__(self, model_path='models/shelter_model_lstm.h5'):
        """Initialize the predictor with trained model and preprocessors"""
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.shelter_names = None
        
        # Load model and preprocessors
        self._load_model_and_preprocessors(model_path)
    
    def _load_model_and_preprocessors(self, model_path):
        """Load the trained model and preprocessors"""
        try:
            # Load model
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
            
            # Load preprocessors
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.shelter_names = joblib.load('models/shelter_names.pkl')
            print("✓ Preprocessors loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model/preprocessors: {e}")
            raise
    
    def predict_occupancy(self, date, shelter_name):
        """Predict occupancy for a specific date and shelter"""
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        # Convert date to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Create features for prediction
        features = self._create_prediction_features(date, shelter_name)
        
        # Make prediction
        prediction = self.model.predict(features.reshape(1, 30, -1), verbose=0)
        
        return max(0, int(prediction[0][0]))  # Ensure non-negative integer
    
    def _create_prediction_features(self, date, shelter_name, sequence_length=30):
        """Create features for prediction"""
        features = []
        
        for i in range(sequence_length):
            # Go back in time to create sequence
            current_date = date - pd.Timedelta(days=sequence_length-i-1)
            
            # Create date features
            year = current_date.year
            month = current_date.month
            day = current_date.day
            day_of_week = current_date.dayofweek
            day_of_year = current_date.dayofyear
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            # Cyclical features
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            day_sin = np.sin(2 * np.pi * day / 31)
            day_cos = np.cos(2 * np.pi * day / 31)
            day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Encode shelter name
            shelter_encoded = self.label_encoders['SHELTER_NAME'].transform([shelter_name])[0]
            
            # Create feature vector (matching training features)
            feature_vector = [
                year, month, day, day_of_week, day_of_year, is_weekend,
                month_sin, month_cos, day_sin, day_cos,
                day_of_week_sin, day_of_week_cos,
                0, shelter_encoded, 0, 0,  # Placeholder encodings
                0, 0, 0, 0, 0, 0, 0  # Placeholder lag features
            ]
            
            features.append(feature_vector)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def get_available_shelters(self):
        """Get list of available shelters"""
        return list(self.shelter_names) if self.shelter_names is not None else []
    
    def predict_date_range(self, start_date, end_date, shelter_name):
        """Predict occupancy for a range of dates"""
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        predictions = []
        
        for date in date_range:
            occupancy = self.predict_occupancy(date, shelter_name)
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'day_of_week': date.strftime('%A'),
                'predicted_occupancy': occupancy
            })
        
        return predictions

def main():
    """Example usage of the ShelterPredictor"""
    try:
        # Initialize predictor
        predictor = ShelterPredictor()
        
        # Get available shelters
        shelters = predictor.get_available_shelters()
        print(f"Available shelters: {len(shelters)}")
        
        # Example prediction
        if shelters:
            shelter_name = shelters[0]
            date = "2024-01-15"
            prediction = predictor.predict_occupancy(date, shelter_name)
            print(f"Predicted occupancy for {shelter_name} on {date}: {prediction}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 