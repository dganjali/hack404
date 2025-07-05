#!/usr/bin/env python3
"""
Simple API for Shelter Occupancy Prediction
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

@app.route('/shelters', methods=['GET'])
def get_shelters():
    """Get list of available shelters"""
    if predictor is None:
        return jsonify({'error': 'Predictor not loaded'}), 500
    
    try:
        shelters = predictor.get_available_shelters()
        return jsonify({
            'shelters': shelters,
            'count': len(shelters)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_occupancy():
    """Predict occupancy for a specific date and shelter"""
    if predictor is None:
        return jsonify({'error': 'Predictor not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        date = data.get('date')
        shelter_name = data.get('shelter_name')
        
        if not date or not shelter_name:
            return jsonify({'error': 'date and shelter_name are required'}), 400
        
        prediction = predictor.predict_occupancy(date, shelter_name)
        
        return jsonify({
            'date': date,
            'shelter_name': shelter_name,
            'predicted_occupancy': prediction
        })
        
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
        
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        shelter_name = data.get('shelter_name')
        
        if not start_date or not end_date or not shelter_name:
            return jsonify({'error': 'start_date, end_date, and shelter_name are required'}), 400
        
        predictions = predictor.predict_date_range(start_date, end_date, shelter_name)
        
        return jsonify({
            'shelter_name': shelter_name,
            'start_date': start_date,
            'end_date': end_date,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 