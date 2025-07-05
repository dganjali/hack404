# Backend - Shelter Occupancy Prediction

This backend contains the organized and cleaned prediction system for shelter occupancy forecasting.

## Structure

```
backend/
└── prediction/              # Main prediction module
    ├── models/             # Trained models and parameters
    ├── shelter_predictor.py # Core prediction class
    ├── api.py              # Flask API interface
    ├── requirements.txt    # Dependencies
    └── README.md          # Documentation
```

## What's Included

### ✅ Trained Model
- **LSTM Model**: Best performing model (MAE: 6.98)
- **Model File**: `models/shelter_model_lstm.h5`
- **Preprocessors**: All necessary encoders and scalers

### ✅ Clean Interface
- **ShelterPredictor**: Simple class for making predictions
- **Flask API**: RESTful API for integration
- **Documentation**: Complete usage examples

### ✅ All Parameters Preserved
- Label encoders for categorical variables
- Feature scaler for normalization
- Shelter names and metadata
- Model architecture and weights

## Quick Test

```bash
cd backend/prediction
python shelter_predictor.py
```

This should output:
```
✓ Model loaded from models/shelter_model_lstm.h5
✓ Preprocessors loaded successfully
Available shelters: 65
Predicted occupancy for COSTI Reception Centre on 2024-01-15: 7
```

## API Usage

Start the API server:
```bash
cd backend/prediction
python api.py
```

Then make requests:
```bash
# Get available shelters
curl http://localhost:5000/shelters

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-01-15", "shelter_name": "COSTI Reception Centre"}'
```

## Model Performance

- **MAE**: 6.98 people
- **RMSE**: 19.81 people  
- **R² Score**: 0.9454
- **Training Data**: 91,216 sequences
- **Test Data**: 22,804 sequences

The system is ready for integration into a larger application! 