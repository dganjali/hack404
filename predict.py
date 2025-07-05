#!/usr/bin/env python3
"""
Shelter Occupancy Prediction Script
Uses the new aggregated prediction pipeline to predict occupancy for any shelter
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shelter_model import ShelterPredictionModel

class ShelterPredictor:
    def __init__(self, model_path='models/shelter_model_lstm.h5', scaler_path='models/scaler_lstm.pkl'):
        """Initialize the predictor with trained model"""
        self.model = ShelterPredictionModel()
        self.scaler = None
        
        # Load model
        try:
            self.model.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Please ensure the model is trained first using train_model.py")
            return
        
        # Load scaler
        try:
            self.scaler = joblib.load(scaler_path)
            self.model.scaler = self.scaler
            print(f"✓ Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"✗ Error loading scaler: {e}")
    
    def predict_occupancy(self, shelter_info, target_date):
        """Predict occupancy for a specific shelter on a specific date"""
        try:
            prediction = self.model.predict_for_shelter(shelter_info, target_date)
            return prediction
        except Exception as e:
            print(f"✗ Error making prediction: {e}")
            return None
    
    def predict_multiple_dates(self, shelter_info, start_date, days_ahead=7):
        """Predict occupancy for multiple consecutive days"""
        predictions = []
        current_date = pd.to_datetime(start_date)
        
        for i in range(days_ahead):
            date_str = current_date.strftime('%Y-%m-%d')
            prediction = self.predict_occupancy(shelter_info, date_str)
            
            if prediction:
                predictions.append(prediction)
            
            current_date += timedelta(days=1)
        
        return predictions
    
    def predict_for_multiple_shelters(self, shelters, target_date):
        """Predict occupancy for multiple shelters on the same date"""
        predictions = []
        
        for shelter in shelters:
            prediction = self.predict_occupancy(shelter, target_date)
            if prediction:
                predictions.append(prediction)
        
        return predictions

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Predict shelter occupancy')
    parser.add_argument('--date', type=str, default=None,
                       help='Target date (YYYY-MM-DD). Defaults to tomorrow.')
    parser.add_argument('--shelter-name', type=str, default='Test Shelter',
                       help='Shelter name')
    parser.add_argument('--capacity', type=int, default=100,
                       help='Shelter maximum capacity')
    parser.add_argument('--days-ahead', type=int, default=1,
                       help='Number of days to predict ahead')
    parser.add_argument('--model-path', type=str, default='models/shelter_model_lstm.h5',
                       help='Path to trained model')
    parser.add_argument('--scaler-path', type=str, default='models/scaler_lstm.pkl',
                       help='Path to scaler')
    
    args = parser.parse_args()
    
    # Set default date to tomorrow if not provided
    if args.date is None:
        tomorrow = datetime.now() + timedelta(days=1)
        args.date = tomorrow.strftime('%Y-%m-%d')
    
    # Initialize predictor
    predictor = ShelterPredictor(args.model_path, args.scaler_path)
    
    # Create shelter info
    shelter_info = {
        'name': args.shelter_name,
        'maxCapacity': args.capacity,
        'address': args.shelter_name,  # Default address - in practice, get from user input
        'postal_code': 'M5S 2P1',  # Default postal code - in practice, get from user input
        'city': 'Toronto',
        'province': 'ON'
    }
    
    print(f"\nPredicting occupancy for {args.shelter_name}")
    print(f"Target date: {args.date}")
    print(f"Max capacity: {args.capacity}")
    print("-" * 50)
    
    if args.days_ahead == 1:
        # Single day prediction
        prediction = predictor.predict_occupancy(shelter_info, args.date)
        
        if prediction:
            print(f"Predicted Occupancy: {prediction['predicted_occupancy']}")
            print(f"Utilization Rate: {prediction['utilization_rate']}%")
            print(f"Max Capacity: {prediction['max_capacity']}")
        else:
            print("Prediction failed")
    
    else:
        # Multiple days prediction
        predictions = predictor.predict_multiple_dates(shelter_info, args.date, args.days_ahead)
        
        if predictions:
            print("Predictions:")
            print(f"{'Date':<12} {'Occupancy':<12} {'Utilization':<12}")
            print("-" * 40)
            
            for pred in predictions:
                print(f"{pred['target_date']:<12} {pred['predicted_occupancy']:<12} {pred['utilization_rate']}%")
        else:
            print("Predictions failed")

def interactive_mode():
    """Interactive prediction mode"""
    print("Shelter Occupancy Prediction - Interactive Mode")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ShelterPredictor()
    
    while True:
        print("\nOptions:")
        print("1. Predict for a single shelter and date")
        print("2. Predict for multiple days")
        print("3. Predict for multiple shelters")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Single prediction
            shelter_name = input("Enter shelter name: ").strip()
            capacity = int(input("Enter shelter capacity: "))
            target_date = input("Enter target date (YYYY-MM-DD): ").strip()
            
            shelter_info = {
                'name': shelter_name,
                'maxCapacity': capacity,
                'address': shelter_name,  # Default address
                'postal_code': 'M5S 2P1',  # Default postal code
                'city': 'Toronto',
                'province': 'ON'
            }
            
            prediction = predictor.predict_occupancy(shelter_info, target_date)
            
            if prediction:
                print(f"\nPrediction Results:")
                print(f"Shelter: {prediction['shelter_name']}")
                print(f"Date: {prediction['target_date']}")
                print(f"Predicted Occupancy: {prediction['predicted_occupancy']}")
                print(f"Max Capacity: {prediction['max_capacity']}")
                print(f"Utilization Rate: {prediction['utilization_rate']}%")
            else:
                print("Prediction failed")
        
        elif choice == '2':
            # Multiple days prediction
            shelter_name = input("Enter shelter name: ").strip()
            capacity = int(input("Enter shelter capacity: "))
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            days_ahead = int(input("Enter number of days to predict: "))
            
            shelter_info = {
                'name': shelter_name,
                'maxCapacity': capacity,
                'address': shelter_name,  # Default address
                'postal_code': 'M5S 2P1',  # Default postal code
                'city': 'Toronto',
                'province': 'ON'
            }
            
            predictions = predictor.predict_multiple_dates(shelter_info, start_date, days_ahead)
            
            if predictions:
                print(f"\nPredictions for {shelter_name}:")
                print(f"{'Date':<12} {'Occupancy':<12} {'Utilization':<12}")
                print("-" * 40)
                
                for pred in predictions:
                    print(f"{pred['target_date']:<12} {pred['predicted_occupancy']:<12} {pred['utilization_rate']}%")
            else:
                print("Predictions failed")
        
        elif choice == '3':
            # Multiple shelters prediction
            target_date = input("Enter target date (YYYY-MM-DD): ").strip()
            num_shelters = int(input("Enter number of shelters: "))
            
            shelters = []
            for i in range(num_shelters):
                name = input(f"Enter shelter {i+1} name: ").strip()
                capacity = int(input(f"Enter shelter {i+1} capacity: "))
                shelters.append({
                    'name': name,
                    'maxCapacity': capacity
                })
            
            predictions = predictor.predict_for_multiple_shelters(shelters, target_date)
            
            if predictions:
                print(f"\nPredictions for {target_date}:")
                print(f"{'Shelter':<20} {'Occupancy':<12} {'Utilization':<12}")
                print("-" * 50)
                
                for pred in predictions:
                    print(f"{pred['shelter_name']:<20} {pred['predicted_occupancy']:<12} {pred['utilization_rate']}%")
            else:
                print("Predictions failed")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        main()
    else:
        # Interactive mode
        interactive_mode() 