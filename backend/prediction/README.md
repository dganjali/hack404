# Shelter Occupancy Prediction System

A deep learning-based system for predicting homeless shelter occupancy in Toronto.

## Overview

This system uses LSTM neural networks trained on historical shelter data (2017-2020) to predict daily occupancy for various shelters across Toronto.

## Model Performance

- **Best Model**: LSTM
- **MAE**: 6.98 people
- **RMSE**: 19.81 people
- **R² Score**: 0.9454

## Files Structure

```
backend/prediction/
├── models/                    # Trained models and preprocessors
│   ├── shelter_model_lstm.h5 # Best performing LSTM model
│   ├── label_encoders.pkl    # Categorical encoders
│   ├── scaler.pkl           # Feature scaler
│   └── shelter_names.pkl    # Available shelter names
├── shelter_predictor.py      # Main prediction class
├── api.py                    # Flask API interface
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from shelter_predictor import ShelterPredictor

# Initialize predictor
predictor = ShelterPredictor()

# Get available shelters
shelters = predictor.get_available_shelters()

# Make a prediction
prediction = predictor.predict_occupancy("2024-01-15", "COSTI Reception Centre")
print(f"Predicted occupancy: {prediction}")
```

### 3. API Usage
```bash
# Start the API server
python api.py

# Health check
curl http://localhost:5000/health

# Get available shelters
curl http://localhost:5000/shelters

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-01-15", "shelter_name": "COSTI Reception Centre"}'
```

## API Endpoints

- `GET /health` - Health check
- `GET /shelters` - Get available shelters
- `POST /predict` - Predict for single date
- `POST /predict_range` - Predict for date range

## Model Details

- **Architecture**: LSTM with 3 layers (128→64→32 units)
- **Input**: 30-day sequence of features
- **Features**: Temporal, categorical, and lagged occupancy
- **Training Data**: 91,216 sequences
- **Test Data**: 22,804 sequences

## Dependencies

- tensorflow-macos==2.15.0
- pandas==2.1.4
- numpy==1.24.3
- scikit-learn==1.3.2
- flask (for API)
- joblib==1.3.2 