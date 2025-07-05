import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta
import holidays

class ShelterPredictionModel:
    def __init__(self, model_type='lstm'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.canada_holidays = holidays.Canada()
        self.feature_names = None
        self.sequence_length = 7
        
    def create_lstm_model(self, input_shape):
        """Create LSTM model for time series prediction"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_conv_lstm_model(self, input_shape):
        """Create CNN-LSTM model for time series prediction"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_attention_model(self, input_shape):
        """Create attention-based model for time series prediction"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers with attention
        lstm1 = LSTM(64, return_sequences=True)(inputs)
        lstm2 = LSTM(32, return_sequences=True)(lstm1)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=8
        )(lstm2, lstm2)
        
        # Combine attention with LSTM output
        concat = Concatenate()([lstm2, attention])
        
        # Final layers
        lstm3 = LSTM(16, return_sequences=False)(concat)
        dense1 = Dense(16, activation='relu')(lstm3)
        dropout = Dropout(0.2)(dense1)
        outputs = Dense(1, activation='linear')(dropout)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model"""
        print(f"Training {self.model_type.upper()} model...")
        
        # Create model based on type
        if self.model_type == 'lstm':
            self.model = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
        elif self.model_type == 'conv_lstm':
            self.model = self.create_conv_lstm_model((X_train.shape[1], X_train.shape[2]))
        elif self.model_type == 'attention':
            self.model = self.create_attention_model((X_train.shape[1], X_train.shape[2]))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Training
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        print("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred.flatten(),
            'actual': y_test
        }
    
    def predict_future(self, last_sequence, days_ahead=7):
        """Predict future occupancy for the next N days"""
        print(f"Predicting occupancy for next {days_ahead} days...")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for day in range(days_ahead):
            # Predict next day
            pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape))
            predictions.append(pred[0][0])
            
            # Update sequence for next prediction
            # This is a simplified approach - in practice, you'd need to update
            # all the features (weather, holidays, etc.) for the next day
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0][0]  # Update occupancy prediction
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def prepare_prediction_features(self, target_date, historical_data=None):
        """Prepare features for a specific date prediction"""
        print(f"Preparing features for {target_date}...")
        
        # Create date range for sequence
        target_dt = pd.to_datetime(target_date)
        start_date = target_dt - timedelta(days=self.sequence_length)
        
        # Generate features for the sequence
        sequence_features = []
        
        for i in range(self.sequence_length):
            current_date = start_date + timedelta(days=i)
            
            # Temporal features
            features = {
                'year': current_date.year,
                'month': current_date.month,
                'day_of_week': current_date.dayofweek,
                'day_of_month': current_date.day,
                'week_of_year': current_date.isocalendar().week,
                'quarter': current_date.quarter,
                'season': self._get_season(current_date.month),
                'is_weekend': int(current_date.dayofweek >= 5),
                'is_month_end': int(current_date.is_month_end),
                'is_month_start': int(current_date.is_month_start),
                'is_holiday': int(current_date in self.canada_holidays),
                'temperature': self._simulate_temperature(current_date),
                'precipitation_mm': self._simulate_precipitation(current_date),
                'weather_severity': self._get_weather_severity(current_date)
            }
            
            # Add historical averages if available
            if historical_data is not None:
                features['dow_avg'] = historical_data.get('dow_avg', {}).get(current_date.dayofweek, 0)
                features['month_avg'] = historical_data.get('month_avg', {}).get(current_date.month, 0)
            else:
                features['dow_avg'] = 0
                features['month_avg'] = 0
            
            sequence_features.append(list(features.values()))
        
        return np.array(sequence_features)
    
    def _get_season(self, month):
        """Get season (1=Winter, 2=Spring, 3=Summer, 4=Fall)"""
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Fall
    
    def _simulate_temperature(self, date):
        """Simulate temperature based on season"""
        season = self._get_season(date.month)
        
        if season == 1:  # Winter
            return np.random.normal(-5, 10)
        elif season == 2:  # Spring
            return np.random.normal(10, 8)
        elif season == 3:  # Summer
            return np.random.normal(25, 8)
        else:  # Fall
            return np.random.normal(15, 8)
    
    def _simulate_precipitation(self, date):
        """Simulate precipitation"""
        season = self._get_season(date.month)
        
        precip_prob = {1: 0.3, 2: 0.4, 3: 0.2, 4: 0.3}[season]
        
        if np.random.random() < precip_prob:
            return np.random.exponential(5)
        return 0
    
    def _get_weather_severity(self, date):
        """Get weather severity (1=mild, 2=moderate, 3=severe)"""
        temp = self._simulate_temperature(date)
        precip = self._simulate_precipitation(date)
        
        if (temp < -10) or (temp > 35) or (precip > 20):
            return 3
        elif (temp < 0) or (temp > 25) or (precip > 10):
            return 2
        else:
            return 1
    
    def predict_for_shelter(self, shelter_info, target_date, historical_data=None):
        """Predict occupancy for a specific shelter on a specific date"""
        print(f"Predicting occupancy for shelter: {shelter_info.get('name', 'Unknown')}")
        print(f"Target date: {target_date}")
        
        # Prepare features
        features = self.prepare_prediction_features(target_date, historical_data)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_reshaped = features.reshape(-1, features.shape[-1])
            features_scaled = self.scaler.transform(features_reshaped)
            features = features_scaled.reshape(features.shape)
        
        # Make prediction
        prediction = self.model.predict(features.reshape(1, *features.shape))
        predicted_occupancy = prediction[0][0]
        
        # Apply shelter-specific scaling if capacity is provided
        if 'maxCapacity' in shelter_info:
            max_capacity = shelter_info['maxCapacity']
            # Scale prediction based on shelter capacity
            # This assumes the model predicts average occupancy per shelter
            scaled_prediction = min(predicted_occupancy * (max_capacity / 100), max_capacity)
            predicted_occupancy = max(0, scaled_prediction)
        
        return {
            'shelter_name': shelter_info.get('name', 'Unknown'),
            'target_date': target_date,
            'predicted_occupancy': round(predicted_occupancy, 0),
            'max_capacity': shelter_info.get('maxCapacity', None),
            'utilization_rate': round((predicted_occupancy / shelter_info.get('maxCapacity', 100)) * 100, 1) if shelter_info.get('maxCapacity') else None
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, save_path=None):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Occupancy')
        plt.ylabel('Predicted Occupancy')
        plt.title('Actual vs Predicted Occupancy')
        plt.grid(True)
        
        # Plot time series
        plt.subplot(1, 2, 2)
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Occupancy')
        plt.title('Time Series Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # This would be used after data preprocessing
    print("Shelter Prediction Model")
    print("This module provides the prediction model for the shelter occupancy system.") 