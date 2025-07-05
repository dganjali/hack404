#!/usr/bin/env python3
"""
Simple API for Shelter Occupancy Prediction with Location Features
"""

from flask import Flask, request, jsonify
from shelter_predictor import ShelterPredictor
import traceback

app = Flask(__name__)

# Initialize predictor
try:
    predictor = ShelterPredictor()
    print("✓ Predictor initialized successfully")
except Exception as e:
    print(f"✗ Error initializing predictor: {e}")
    predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'predictor_loaded': predictor is not None
    })

@app.route('/predict', methods=['POST'])
def predict_occupancy():
    """Predict occupancy for a specific shelter on a specific date"""
    if predictor is None:
        return jsonify({'error': 'Predictor not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        shelter_info = data.get('shelter_info')
        target_date = data.get('target_date')
        
        if not shelter_info or not target_date:
            return jsonify({'error': 'shelter_info and target_date are required'}), 400
        
        # Ensure shelter_info has required fields
        if not shelter_info.get('name'):
            return jsonify({'error': 'shelter_info must include name'}), 400
        
        prediction = predictor.predict_occupancy(shelter_info, target_date)
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_range', methods=['POST'])
def predict_range():
    """Predict occupancy for a date range"""
    if predictor is None:
        return jsonify({'error': 'Predictor not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        shelter_info = data.get('shelter_info')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not shelter_info or not start_date or not end_date:
            return jsonify({'error': 'shelter_info, start_date, and end_date are required'}), 400
        
        # Ensure shelter_info has required fields
        if not shelter_info.get('name'):
            return jsonify({'error': 'shelter_info must include name'}), 400
        
        predictions = predictor.predict_date_range(shelter_info, start_date, end_date)
        
        return jsonify({
            'shelter_info': shelter_info,
            'start_date': start_date,
            'end_date': end_date,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sectors', methods=['GET'])
def get_sectors():
    """Get available Toronto sectors and their information"""
    sectors = {
        'downtown_core': {
            'name': 'Downtown Core',
            'description': 'Financial district, entertainment district, university area',
            'postal_codes': ['M5A', 'M5B', 'M5C', 'M5E', 'M5G', 'M5H', 'M5J', 'M5K', 'M5L', 'M5M', 'M5N', 'M5P', 'M5R', 'M5S', 'M5T', 'M5V', 'M5W', 'M5X', 'M5Y', 'M5Z']
        },
        'east_end': {
            'name': 'East End',
            'description': 'Scarborough, East York, Beaches',
            'postal_codes': ['M1B', 'M1C', 'M1E', 'M1G', 'M1H', 'M1J', 'M1K', 'M1L', 'M1M', 'M1N', 'M1P', 'M1R', 'M1S', 'M1T', 'M1V', 'M1W', 'M1X']
        },
        'west_end': {
            'name': 'West End',
            'description': 'West Toronto, Parkdale, High Park, Junction',
            'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S']
        },
        'north_end': {
            'name': 'North End',
            'description': 'North York, York, Don Mills, Lawrence Park',
            'postal_codes': ['M2H', 'M2J', 'M2K', 'M2L', 'M2M', 'M2N', 'M2P', 'M2R', 'M3A', 'M3B', 'M3C', 'M3H', 'M3J', 'M3K', 'M3L', 'M3M', 'M3N', 'M4A', 'M4B', 'M4C', 'M4E', 'M4G', 'M4H', 'M4J', 'M4K', 'M4L', 'M4M', 'M4N', 'M4P', 'M4R', 'M4S', 'M4T', 'M4V', 'M4W', 'M4X', 'M4Y']
        },
        'etobicoke': {
            'name': 'Etobicoke',
            'description': 'Etobicoke, Rexdale, Humber Bay',
            'postal_codes': ['M8V', 'M8W', 'M8X', 'M8Y', 'M8Z', 'M9A', 'M9B', 'M9C', 'M9P', 'M9R', 'M9V', 'M9W']
        },
        'york': {
            'name': 'York',
            'description': 'York, Weston, Mount Dennis',
            'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S']
        }
    }
    
    return jsonify({
        'sectors': sectors,
        'count': len(sectors)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 