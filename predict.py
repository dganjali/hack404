#!/usr/bin/env python3
"""
Prediction script for the Shelter Occupancy Prediction Model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os

from shelter_model import ShelterPredictor

def load_available_shelters():
    """Load available shelter names from the preprocessors"""
    try:
        import joblib
        shelter_names = joblib.load('models/shelter_names.pkl')
        return shelter_names
    except:
        # Fallback to a sample list if preprocessors not available
        return [
            "COSTI Reception Centre",
            "Christie Ossington Men's Hostel", 
            "Christie Refugee Welcome Centre",
            "Birchmount Residence",
            "Birkdale Residence",
            "Downsview Dells",
            "Family Residence"
        ]

def predict_single_date(date, shelter_name, model_path='models/shelter_model_lstm.h5'):
    """Predict occupancy for a single date and shelter"""
    try:
        predictor = ShelterPredictor(model_path)
        
        # Convert date string to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Make prediction
        predicted_occupancy = predictor.predict_occupancy(date, shelter_name)
        
        return predicted_occupancy
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def predict_date_range(start_date, end_date, shelter_name, model_path='models/shelter_model_lstm.h5'):
    """Predict occupancy for a range of dates"""
    try:
        predictor = ShelterPredictor(model_path)
        
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        predictions = []
        for date in date_range:
            occupancy = predictor.predict_occupancy(date, shelter_name)
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'day_of_week': date.strftime('%A'),
                'predicted_occupancy': occupancy
            })
        
        return predictions
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def display_predictions(predictions, shelter_name):
    """Display predictions in a formatted table"""
    if not predictions:
        print("No predictions available.")
        return
    
    print(f"\nPredictions for {shelter_name}")
    print("="*60)
    print(f"{'Date':<12} {'Day':<12} {'Predicted Occupancy':<20}")
    print("-"*60)
    
    for pred in predictions:
        print(f"{pred['date']:<12} {pred['day_of_week']:<12} {pred['predicted_occupancy']:<20}")
    
    # Calculate statistics
    occupancies = [p['predicted_occupancy'] for p in predictions]
    avg_occupancy = np.mean(occupancies)
    max_occupancy = max(occupancies)
    min_occupancy = min(occupancies)
    
    print("-"*60)
    print(f"Average occupancy: {avg_occupancy:.1f}")
    print(f"Maximum occupancy: {max_occupancy}")
    print(f"Minimum occupancy: {min_occupancy}")

def interactive_mode():
    """Run in interactive mode"""
    print("Shelter Occupancy Prediction - Interactive Mode")
    print("="*50)
    
    # Load available shelters
    shelters = load_available_shelters()
    
    print("\nAvailable shelters:")
    for i, shelter in enumerate(shelters, 1):
        print(f"{i}. {shelter}")
    
    while True:
        print("\n" + "-"*50)
        print("Options:")
        print("1. Predict for a single date")
        print("2. Predict for a date range")
        print("3. List available shelters")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Single date prediction
            try:
                shelter_idx = int(input(f"Enter shelter number (1-{len(shelters)}): ")) - 1
                if 0 <= shelter_idx < len(shelters):
                    shelter_name = shelters[shelter_idx]
                    date_str = input("Enter date (YYYY-MM-DD): ")
                    
                    prediction = predict_single_date(date_str, shelter_name)
                    if prediction is not None:
                        print(f"\nPredicted occupancy for {shelter_name} on {date_str}: {prediction}")
                else:
                    print("Invalid shelter number.")
            except ValueError:
                print("Invalid input.")
        
        elif choice == '2':
            # Date range prediction
            try:
                shelter_idx = int(input(f"Enter shelter number (1-{len(shelters)}): ")) - 1
                if 0 <= shelter_idx < len(shelters):
                    shelter_name = shelters[shelter_idx]
                    start_date = input("Enter start date (YYYY-MM-DD): ")
                    end_date = input("Enter end date (YYYY-MM-DD): ")
                    
                    predictions = predict_date_range(start_date, end_date, shelter_name)
                    if predictions:
                        display_predictions(predictions, shelter_name)
                else:
                    print("Invalid shelter number.")
            except ValueError:
                print("Invalid input.")
        
        elif choice == '3':
            print("\nAvailable shelters:")
            for i, shelter in enumerate(shelters, 1):
                print(f"{i}. {shelter}")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Shelter Occupancy Prediction')
    parser.add_argument('--date', type=str, help='Date for prediction (YYYY-MM-DD)')
    parser.add_argument('--shelter', type=str, help='Shelter name')
    parser.add_argument('--start-date', type=str, help='Start date for range prediction')
    parser.add_argument('--end-date', type=str, help='End date for range prediction')
    parser.add_argument('--model', type=str, default='models/shelter_model_lstm.h5', 
                       help='Path to trained model')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--list-shelters', action='store_true',
                       help='List available shelters')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        print("Please train the model first using train_model.py")
        sys.exit(1)
    
    # List shelters
    if args.list_shelters:
        shelters = load_available_shelters()
        print("Available shelters:")
        for i, shelter in enumerate(shelters, 1):
            print(f"{i}. {shelter}")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Single date prediction
    if args.date and args.shelter:
        prediction = predict_single_date(args.date, args.shelter, args.model)
        if prediction is not None:
            print(f"Predicted occupancy for {args.shelter} on {args.date}: {prediction}")
        return
    
    # Date range prediction
    if args.start_date and args.end_date and args.shelter:
        predictions = predict_date_range(args.start_date, args.end_date, args.shelter, args.model)
        if predictions:
            display_predictions(predictions, args.shelter)
        return
    
    # If no valid arguments, show help
    print("No valid arguments provided. Use --help for usage information.")
    print("Or use --interactive for interactive mode.")

if __name__ == "__main__":
    main() 