# ShelterFlow Webapp Update: Location-Based Predictions

## Overview

The ShelterFlow webapp has been updated to use the new location-based prediction model with sector-specific features. This update integrates the trained machine learning model with location features to provide more accurate occupancy predictions.

## New Features

### 1. Location-Based Predictions
- **Sector Mapping**: Shelters are automatically mapped to Toronto sectors based on postal codes
- **Socioeconomic Indicators**: Each sector includes income, population density, transit accessibility, crime rate, and homelessness rate
- **Enhanced Features**: Temporal features (weather, holidays, seasons) combined with location features

### 2. Updated Webapp Interface
- **Postal Code Field**: Added postal code input for shelters to enable location-based predictions
- **Sector Display**: Shows the mapped sector for each shelter in the dashboard
- **Enhanced Predictions**: More accurate predictions using the trained model

### 3. Toronto Sectors
The system maps shelters to these Toronto sectors:
- **Downtown Core**: Financial district, entertainment district, university area
- **East End**: Scarborough, East York, Beaches
- **West End**: West Toronto, Parkdale, High Park, Junction
- **North End**: North York, York, Don Mills, Lawrence Park
- **Etobicoke**: Etobicoke, Rexdale, Humber Bay
- **York**: York, Weston, Mount Dennis

## Technical Changes

### Backend Updates
1. **Python Prediction API** (`backend/prediction/`):
   - Updated `shelter_predictor.py` to use location features
   - New sector mapping based on postal codes
   - Enhanced feature engineering with temporal and location features
   - Updated API endpoints to accept shelter information

2. **Node.js Server** (`backend/server/`):
   - Integrated with Python prediction API
   - Added postal code field to shelter management
   - Updated dashboard to display sector information
   - Added axios dependency for API communication

### Frontend Updates
1. **Dashboard Interface**:
   - Added postal code field in shelter forms
   - Display sector information in shelter cards and table
   - Enhanced prediction display with sector details

2. **JavaScript Updates**:
   - Updated form handling to include postal codes
   - Enhanced shelter display with sector information
   - Improved error handling for API communication

## How to Use

### Starting the Webapp
```bash
# Make the start script executable
chmod +x start_webapp.sh

# Start both servers
./start_webapp.sh
```

The script will:
1. Check for Python 3.11 and required dependencies
2. Start the Python prediction API on port 5000
3. Start the Node.js webapp server on port 3000
4. Verify both services are running

### Adding Shelters with Location Data
1. Navigate to the dashboard
2. Click "Add Shelter"
3. Fill in shelter details including:
   - **Name**: Shelter name
   - **Address**: Full address
   - **Postal Code**: Toronto postal code (e.g., M5J 2T3)
   - **Maximum Capacity**: Total bed capacity
4. The system will automatically:
   - Map the shelter to the appropriate Toronto sector
   - Use sector-specific socioeconomic indicators
   - Generate location-based predictions

### Viewing Predictions
- **Dashboard Overview**: Shows predicted occupancy for all shelters
- **Shelter Cards**: Display sector information and predictions
- **Shelter Table**: Includes sector column with detailed information
- **Individual Predictions**: Access detailed prediction data via API

## API Endpoints

### Python Prediction API (Port 5000)
- `GET /health` - Health check
- `POST /predict` - Single prediction with shelter info
- `POST /predict_range` - Date range predictions
- `GET /sectors` - Available Toronto sectors

### Node.js Webapp API (Port 3000)
- All existing endpoints with enhanced prediction integration
- Updated to use Python prediction API for accurate forecasts

## Model Features

The updated prediction model uses:
- **Temporal Features**: Year, month, day, season, holidays, weather
- **Location Features**: Sector encoding, income, population density, transit accessibility
- **Socioeconomic Indicators**: Crime rate, homelessness rate by sector
- **Sequence Modeling**: 7-day sequence for LSTM predictions

## Troubleshooting

### Common Issues
1. **Python 3.11 not found**: Install using `pyenv install 3.11.0`
2. **Model file missing**: Copy `best_model.h5` to `backend/prediction/models/`
3. **Dependencies missing**: Run `pip3.11 install -r backend/prediction/requirements.txt`
4. **API connection failed**: Check if Python API is running on port 5000

### Fallback Behavior
- If Python API is unavailable, the system falls back to simulated predictions
- Error messages are logged for debugging
- Frontend gracefully handles API failures

## Performance Improvements
- **Faster Predictions**: Optimized feature engineering
- **Better Accuracy**: Location-aware predictions
- **Enhanced UX**: Real-time sector information display
- **Robust Error Handling**: Graceful fallbacks and user feedback

## Next Steps
- Add more detailed sector analytics
- Implement historical data integration
- Add weather API integration for real-time weather data
- Expand to other Canadian cities 