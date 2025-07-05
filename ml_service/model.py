import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from datetime import datetime, timedelta

class ShelterPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(ShelterPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ShelterMLService:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.shelter_data = None
        self.features_data = None
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the shelter data"""
        try:
            # Load shelter data
            with open('../data/real_shelters.json', 'r') as f:
                self.shelter_data = json.load(f)
            
            # Load features data
            with open('../data/real_features.json', 'r') as f:
                self.features_data = json.load(f)
            
            print(f"Loaded {len(self.shelter_data)} shelters and {len(self.features_data)} feature records")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_features(self, data, sequence_length=30):
        """Prepare features for the model"""
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Select features for prediction
        feature_columns = [
            'total_occupancy', 'avg_occupancy', 'utilization_rate',
            'day_of_week', 'month', 'year', 'is_weekend',
            'is_winter', 'is_summer', 'is_spring', 'is_fall'
        ]
        
        # Ensure all columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Prepare features
        features = df[feature_columns].values
        
        # Scale features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            features = self.feature_scaler.fit_transform(features)
        else:
            features = self.feature_scaler.transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(df['total_occupancy'].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model"""
        if not self.load_data():
            return False
        
        # Prepare data
        X, y = self.prepare_features(self.features_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = ShelterPredictor(input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test)
                val_loss = criterion(val_outputs.squeeze(), y_test)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # Save model
        self.save_model()
        
        return True
    
    def save_model(self):
        """Save the trained model"""
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'shelter_predictor.pth'))
            
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, os.path.join(self.model_dir, 'feature_scaler.pkl'))
        
        print("Model saved successfully")
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.path.join(self.model_dir, 'shelter_predictor.pth')
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print("Model files not found. Training new model...")
                return self.train_model()
            
            # Load data first
            if not self.load_data():
                return False
            
            # Load scaler
            self.feature_scaler = joblib.load(scaler_path)
            
            # Initialize and load model
            X, _ = self.prepare_features(self.features_data[:50])  # Just to get input size
            input_size = X.shape[2]
            
            self.model = ShelterPredictor(input_size).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_shelter(self, shelter_id, days=7):
        """Predict occupancy for a specific shelter"""
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # Get shelter data
            shelter = next((s for s in self.shelter_data if s['id'] == shelter_id), None)
            if not shelter:
                return None
            
            # Prepare recent data for prediction
            recent_data = self.features_data[-30:]  # Last 30 days
            X, _ = self.prepare_features(recent_data)
            
            # Make prediction
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X[-1:]).to(self.device)
                prediction = self.model(X_tensor)
                predicted_occupancy = prediction.item()
            
            # Generate future predictions
            future_predictions = []
            current_features = X[-1].copy()
            
            for day in range(days):
                # Update date features
                current_date = datetime.now() + timedelta(days=day)
                current_features[3] = current_date.weekday()  # day_of_week
                current_features[4] = current_date.month      # month
                current_features[5] = current_date.year       # year
                current_features[6] = 1 if current_date.weekday() >= 5 else 0  # is_weekend
                
                # Seasonal features
                month = current_date.month
                current_features[7] = 1 if month in [12, 1, 2] else 0  # is_winter
                current_features[8] = 1 if month in [6, 7, 8] else 0   # is_summer
                current_features[9] = 1 if month in [3, 4, 5] else 0   # is_spring
                current_features[10] = 1 if month in [9, 10, 11] else 0  # is_fall
                
                # Make prediction
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(current_features.reshape(1, 1, -1)).to(self.device)
                    prediction = self.model(X_tensor)
                    future_predictions.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'predicted_occupancy': max(0, prediction.item()),
                        'utilization_rate': min(1.0, max(0, prediction.item() / shelter['capacity']))
                    })
                
                # Update features for next prediction
                current_features[0] = prediction.item()  # total_occupancy
                current_features[1] = prediction.item() / len(self.shelter_data)  # avg_occupancy
                current_features[2] = min(1.0, prediction.item() / shelter['capacity'])  # utilization_rate
            
            return {
                'shelter_id': shelter_id,
                'shelter_name': shelter['name'],
                'current_occupancy': shelter['current_beds'],
                'capacity': shelter['capacity'],
                'predictions': future_predictions
            }
            
        except Exception as e:
            print(f"Error predicting for shelter {shelter_id}: {e}")
            return None
    
    def predict_overview(self, days=7):
        """Predict overview for all shelters"""
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            overview_predictions = []
            
            for shelter in self.shelter_data[:10]:  # Limit to first 10 shelters for performance
                prediction = self.predict_shelter(shelter['id'], days)
                if prediction:
                    overview_predictions.append(prediction)
            
            return {
                'total_shelters': len(overview_predictions),
                'prediction_days': days,
                'predictions': overview_predictions
            }
            
        except Exception as e:
            print(f"Error predicting overview: {e}")
            return None

if __name__ == "__main__":
    # Test the model
    service = ShelterMLService()
    service.train_model(epochs=50) 