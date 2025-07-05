import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

layers = keras.layers
models = keras.models
callbacks = keras.callbacks

class ShelterOccupancyModel:
    def __init__(self, sequence_length=30, n_features=25, n_shelters=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_shelters = n_shelters
        self.model = None
        
    def build_model(self):
        """Build the deep learning model"""
        
        # Input layer
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers with dropout
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(input_layer)
        lstm2 = layers.LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        lstm3 = layers.LSTM(32, return_sequences=False, dropout=0.2)(lstm2)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(lstm3)
        dropout1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        output_layer = layers.Dense(1, activation='linear')(dropout2)
        
        # Create model
        self.model = models.Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def build_attention_model(self):
        """Build model with attention mechanism"""
        
        # Input layer
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(input_layer)
        lstm2 = layers.LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        
        # Attention mechanism
        attention = layers.Attention()([lstm2, lstm2])
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(pooled)
        dropout1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        output_layer = layers.Dense(1, activation='linear')(dropout2)
        
        # Create model
        self.model = models.Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def build_conv_lstm_model(self):
        """Build model with 1D CNN + LSTM"""
        
        # Input layer
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # 1D CNN layers
        conv1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        conv2 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(conv1)
        maxpool = layers.MaxPooling1D(pool_size=2)(conv2)
        
        # LSTM layers
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(maxpool)
        lstm2 = layers.LSTM(64, return_sequences=False, dropout=0.2)(lstm1)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(lstm2)
        dropout1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        output_layer = layers.Dense(1, activation='linear')(dropout2)
        
        # Create model
        self.model = models.Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def get_callbacks(self, model_save_path='models/shelter_model.h5'):
        """Get training callbacks"""
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        return callbacks_list
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, model_type='lstm'):
        """Train the model"""
        
        if model_type == 'attention':
            self.build_attention_model()
        elif model_type == 'conv_lstm':
            self.build_conv_lstm_model()
        else:
            self.build_model()
        
        callbacks_list = self.get_callbacks()
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return {'loss': loss, 'mae': mae}
    
    def save_model(self, filepath='models/shelter_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/shelter_model.h5'):
        """Load a trained model"""
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")

class ShelterPredictor:
    def __init__(self, model_path='models/shelter_model.h5', preprocessor_path='models'):
        self.model = None
        self.preprocessor = None
        self.load_model_and_preprocessor(model_path, preprocessor_path)
    
    def load_model_and_preprocessor(self, model_path, preprocessor_path):
        """Load the trained model and preprocessors"""
        try:
            # Load model
            self.model = models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Load preprocessors
            from data_preprocessing import ShelterDataPreprocessor
            self.preprocessor = ShelterDataPreprocessor()
            self.preprocessor.load_preprocessors(preprocessor_path)
            print(f"Preprocessors loaded from {preprocessor_path}")
            
        except Exception as e:
            print(f"Error loading model/preprocessors: {e}")
            print("Please ensure the model is trained first.")
    
    def predict_occupancy(self, date, shelter_name, sequence_length=30):
        """Predict occupancy for a specific date and shelter"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model or preprocessors not loaded properly")
        
        # Convert date to datetime
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Create features for the prediction
        features = self._create_prediction_features(date, shelter_name, sequence_length)
        
        # Make prediction
        prediction = self.model.predict(features.reshape(1, sequence_length, -1))
        
        return max(0, int(prediction[0][0]))  # Ensure non-negative integer
    
    def _create_prediction_features(self, date, shelter_name, sequence_length):
        """Create features for prediction"""
        # This is a simplified version - in practice, you'd need historical data
        # to create the full sequence. For now, we'll create synthetic features
        
        features = []
        
        for i in range(sequence_length):
            # Go back in time to create sequence
            current_date = date - pd.Timedelta(days=sequence_length-i-1)
            
            # Create date features
            year = current_date.year
            month = current_date.month
            day = current_date.day
            day_of_week = current_date.dayofweek
            day_of_year = current_date.dayofyear
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            # Cyclical features
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            day_sin = np.sin(2 * np.pi * day / 31)
            day_cos = np.cos(2 * np.pi * day / 31)
            day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Encode shelter name
            shelter_encoded = self.preprocessor.label_encoders['SHELTER_NAME'].transform([shelter_name])[0]
            
            # Create feature vector (simplified - you'd need to match the training features exactly)
            feature_vector = [
                year, month, day, day_of_week, day_of_year, is_weekend,
                month_sin, month_cos, day_sin, day_cos,
                day_of_week_sin, day_of_week_cos,
                0, shelter_encoded, 0, 0,  # Placeholder encodings
                0, 0, 0, 0, 0, 0, 0  # Placeholder lag features
            ]
            
            features.append(feature_vector)
        
        # Scale features
        features_scaled = self.preprocessor.scaler.transform(features)
        
        return features_scaled 