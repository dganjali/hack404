#!/usr/bin/env python3
"""
Python script for making single predictions that can be called from Node.js
"""

import sys
import json
from shelter_predictor import ShelterPredictor

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_single.py <date> <shelter_name>")
        sys.exit(1)
    
    date = sys.argv[1]
    shelter_name = sys.argv[2]
    
    try:
        # Initialize predictor
        predictor = ShelterPredictor()
        
        # Make prediction
        prediction = predictor.predict_occupancy(date, shelter_name)
        
        # Print result (Node.js will capture this)
        print(prediction)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 