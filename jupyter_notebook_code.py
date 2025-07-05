# ============================================================================
# JUPYTER NOTEBOOK CODE FOR SHELTER PREDICTION WITH LOCATION FEATURES
# ============================================================================

# CELL 1: Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os
from datetime import datetime, timedelta
import holidays
import warnings
warnings.filterwarnings('ignore')

# GPU Configuration
print("Configuring GPU for TensorFlow...")

# Check if GPU is available
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configure GPU memory growth to avoid memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled successfully!")
        
        # Optional: Set memory limit (uncomment if needed)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        # )
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")

# Set mixed precision for better performance
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled for better performance!")
except:
    print("Mixed precision not available, using default precision.")

print("All imports and GPU configuration completed successfully!")

# CELL 2: Location Features Pipeline
class TorontoSectorMapper:
    def __init__(self):
        # Define Toronto sectors based on postal code areas
        self.toronto_sectors = {
            'downtown_core': {
                'name': 'Downtown Core',
                'postal_codes': ['M5A', 'M5B', 'M5C', 'M5E', 'M5G', 'M5H', 'M5J', 'M5K', 'M5L', 'M5M', 'M5N', 'M5P', 'M5R', 'M5S', 'M5T', 'M5V', 'M5W', 'M5X', 'M5Y', 'M5Z'],
                'description': 'Financial district, entertainment district, university area'
            },
            'east_end': {
                'name': 'East End',
                'postal_codes': ['M1B', 'M1C', 'M1E', 'M1G', 'M1H', 'M1J', 'M1K', 'M1L', 'M1M', 'M1N', 'M1P', 'M1R', 'M1S', 'M1T', 'M1V', 'M1W', 'M1X'],
                'description': 'Scarborough, East York, Beaches'
            },
            'west_end': {
                'name': 'West End',
                'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S'],
                'description': 'West Toronto, Parkdale, High Park, Junction'
            },
            'north_end': {
                'name': 'North End',
                'postal_codes': ['M2H', 'M2J', 'M2K', 'M2L', 'M2M', 'M2N', 'M2P', 'M2R', 'M3A', 'M3B', 'M3C', 'M3H', 'M3J', 'M3K', 'M3L', 'M3M', 'M3N', 'M4A', 'M4B', 'M4C', 'M4E', 'M4G', 'M4H', 'M4J', 'M4K', 'M4L', 'M4M', 'M4N', 'M4P', 'M4R', 'M4S', 'M4T', 'M4V', 'M4W', 'M4X', 'M4Y'],
                'description': 'North York, York, Don Mills, Lawrence Park'
            },
            'etobicoke': {
                'name': 'Etobicoke',
                'postal_codes': ['M8V', 'M8W', 'M8X', 'M8Y', 'M8Z', 'M9A', 'M9B', 'M9C', 'M9P', 'M9R', 'M9V', 'M9W'],
                'description': 'Etobicoke, Rexdale, Humber Bay'
            },
            'york': {
                'name': 'York',
                'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S'],
                'description': 'York, Weston, Mount Dennis'
            }
        }
        
        # Sector socioeconomic indicators
        self.sector_indicators = {
            'downtown_core': {'avg_income': 85000, 'population_density': 8500, 'transit_accessibility': 0.95, 'crime_rate': 0.3, 'homelessness_rate': 0.8},
            'east_end': {'avg_income': 65000, 'population_density': 4200, 'transit_accessibility': 0.75, 'crime_rate': 0.4, 'homelessness_rate': 0.6},
            'west_end': {'avg_income': 72000, 'population_density': 5800, 'transit_accessibility': 0.85, 'crime_rate': 0.35, 'homelessness_rate': 0.7},
            'north_end': {'avg_income': 95000, 'population_density': 3200, 'transit_accessibility': 0.65, 'crime_rate': 0.2, 'homelessness_rate': 0.3},
            'etobicoke': {'avg_income': 78000, 'population_density': 2800, 'transit_accessibility': 0.55, 'crime_rate': 0.25, 'homelessness_rate': 0.4},
            'york': {'avg_income': 68000, 'population_density': 3800, 'transit_accessibility': 0.70, 'crime_rate': 0.45, 'homelessness_rate': 0.5}
        }
        
        # Initialize sector encoder with all known sectors plus 'unknown'
        all_sectors = list(self.toronto_sectors.keys()) + ['unknown']
        self.sector_encoder = LabelEncoder()
        self.sector_encoder.fit(all_sectors)
        
    def get_sector_from_postal_code(self, postal_code):
        if pd.isna(postal_code) or postal_code == '':
            return 'unknown'
        fsa = str(postal_code).strip()[:3].upper()
        for sector_id, sector_info in self.toronto_sectors.items():
            if fsa in sector_info['postal_codes']:
                return sector_id
        return 'unknown'
    
    def assign_shelter_to_sector(self, shelter_data):
        df = shelter_data.copy()
        df['sector'] = df.apply(lambda row: self.get_sector_from_postal_code(row['SHELTER_POSTAL_CODE']) if pd.notna(row['SHELTER_POSTAL_CODE']) else 'unknown', axis=1)
        
        # Handle any sectors that might not be in the encoder
        unique_sectors = df['sector'].unique()
        for sector in unique_sectors:
            if sector not in self.sector_encoder.classes_:
                print(f"Warning: Unknown sector '{sector}' found, mapping to 'unknown'")
                df.loc[df['sector'] == sector, 'sector'] = 'unknown'
        
        df['sector_encoded'] = self.sector_encoder.transform(df['sector'])
        
        for indicator in ['avg_income', 'population_density', 'transit_accessibility', 'crime_rate', 'homelessness_rate']:
            df[f'sector_{indicator}'] = df['sector'].map({sector: self.sector_indicators[sector][indicator] for sector in self.sector_indicators.keys()}).fillna(0)
        
        return df
    
    def create_sector_features(self, df):
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)
        
        sector_stats = df.groupby('sector').agg({
            'OCCUPANCY': ['mean', 'std', 'min', 'max'],
            'SHELTER_NAME': 'nunique'
        }).reset_index()
        
        sector_stats.columns = ['sector', 'sector_avg_occupancy', 'sector_std_occupancy', 'sector_min_occupancy', 'sector_max_occupancy', 'shelters_in_sector']
        df = df.merge(sector_stats, on='sector', how='left')
        
        df['sector_occupancy_ratio'] = df['OCCUPANCY'] / df['sector_avg_occupancy']
        df['sector_occupancy_zscore'] = (df['OCCUPANCY'] - df['sector_avg_occupancy']) / df['sector_std_occupancy']
        
        return df
    
    def get_sector_for_new_shelter(self, address, postal_code=None, city='Toronto', province='ON'):
        if postal_code:
            sector_id = self.get_sector_from_postal_code(postal_code)
        else:
            sector_id = 'unknown'
        
        # Ensure sector_id is in the encoder classes
        if sector_id not in self.sector_encoder.classes_:
            print(f"Warning: Unknown sector '{sector_id}', mapping to 'unknown'")
            sector_id = 'unknown'
        
        sector_info = {
            'sector_id': sector_id,
            'sector_name': self.toronto_sectors.get(sector_id, {}).get('name', 'Unknown'),
            'sector_description': self.toronto_sectors.get(sector_id, {}).get('description', 'Unknown'),
            'socioeconomic_indicators': self.sector_indicators.get(sector_id, {}),
            'sector_encoded': self.sector_encoder.transform([sector_id])[0]
        }
        
        return sector_info

class LocationFeaturePipeline:
    def __init__(self):
        self.sector_mapper = TorontoSectorMapper()
        self.feature_columns = []
        
    def process_location_features(self, df):
        df = self.sector_mapper.assign_shelter_to_sector(df)
        df = self.sector_mapper.create_sector_features(df)
        self.feature_columns = [col for col in df.columns if any(prefix in col for prefix in ['sector_', 'shelters_in_sector', 'sector_occupancy'])]
        return df
    
    def get_location_features_for_prediction(self, shelter_info):
        sector_info = self.sector_mapper.get_sector_for_new_shelter(
            address=shelter_info.get('address', ''),
            postal_code=shelter_info.get('postal_code'),
            city=shelter_info.get('city', 'Toronto'),
            province=shelter_info.get('province', 'ON')
        )
        
        features = {
            'sector_id': sector_info['sector_id'],
            'sector_encoded': sector_info['sector_encoded'],
            'sector_avg_income': sector_info['socioeconomic_indicators'].get('avg_income', 0),
            'sector_population_density': sector_info['socioeconomic_indicators'].get('population_density', 0),
            'sector_transit_accessibility': sector_info['socioeconomic_indicators'].get('transit_accessibility', 0),
            'sector_crime_rate': sector_info['socioeconomic_indicators'].get('crime_rate', 0),
            'sector_homelessness_rate': sector_info['socioeconomic_indicators'].get('homelessness_rate', 0)
        }
        
        return features

print("Location features pipeline created!")

# CELL 3: Data Preprocessing
class ShelterDataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.shelter_names = None
        self.canada_holidays = holidays.Canada()
        self.location_pipeline = LocationFeaturePipeline()
        
    def load_data(self):
        all_data = []
        for year in [2017, 2018, 2019, 2020]:
            file_path = os.path.join(self.data_dir, f'Daily shelter occupancy {year}.csv')
            if os.path.exists(file_path):
                print(f"Loading {file_path}...")
                df = pd.read_csv(file_path)
                all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset shape: {combined_df.shape}")
        return combined_df
    
    def preprocess_data(self, df):
        df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'], errors='coerce', infer_datetime_format=True)
        df = df.dropna(subset=['OCCUPANCY_DATE'])
        
        # Extract date features
        df['year'] = df['OCCUPANCY_DATE'].dt.year
        df['month'] = df['OCCUPANCY_DATE'].dt.month
        df['day'] = df['OCCUPANCY_DATE'].dt.day
        df['day_of_week'] = df['OCCUPANCY_DATE'].dt.dayofweek
        df['day_of_year'] = df['OCCUPANCY_DATE'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Process location features
        df = self.location_pipeline.process_location_features(df)
        
        # Encode categorical variables
        categorical_cols = ['ORGANIZATION_NAME', 'SHELTER_NAME', 'SECTOR', 'PROGRAM_NAME']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        self.shelter_names = df['SHELTER_NAME'].unique()
        return df
    
    def create_sequences(self, df, sequence_length=30):
        sequences = []
        targets = []
        
        df_sorted = df.sort_values(['SHELTER_NAME', 'OCCUPANCY_DATE'])
        
        for shelter in df['SHELTER_NAME'].unique():
            shelter_data = df_sorted[df_sorted['SHELTER_NAME'] == shelter].copy()
            
            if len(shelter_data) < sequence_length + 1:
                continue
                
            feature_cols = [
                'year', 'month', 'day', 'day_of_week', 'day_of_year', 'is_weekend',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 
                'day_of_week_sin', 'day_of_week_cos',
                'ORGANIZATION_NAME_encoded', 'SHELTER_NAME_encoded', 
                'SECTOR_encoded', 'PROGRAM_NAME_encoded'
            ]
            
            # Add location features
            location_features = [col for col in shelter_data.columns if any(prefix in col for prefix in ['sector_', 'shelters_in_sector', 'sector_occupancy'])]
            feature_cols.extend(location_features)
            
            # Add lagged occupancy features
            for lag in range(1, 8):
                shelter_data[f'occupancy_lag_{lag}'] = shelter_data['OCCUPANCY'].shift(lag)
            
            lag_cols = [col for col in shelter_data.columns if 'lag_' in col]
            shelter_data[lag_cols] = shelter_data[lag_cols].fillna(0)
            feature_cols.extend(lag_cols)
            
            # Create sequences
            for i in range(sequence_length, len(shelter_data)):
                sequence = shelter_data.iloc[i-sequence_length:i][feature_cols].values
                target = shelter_data.iloc[i]['OCCUPANCY']
                sequences.append(sequence)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, sequence_length=30):
        print("Loading data...")
        df = self.load_data()
        
        print("Preprocessing data...")
        df = self.preprocess_data(df)
        
        print("Creating sequences...")
        X, y = self.create_sequences(df, sequence_length)
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        n_features = X_train.shape[-1]
        original_train_shape = X_train.shape
        original_test_shape = X_test.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        X_train = X_train_scaled.reshape(original_train_shape)
        X_test = X_test_scaled.reshape(original_test_shape)
        
        return X_train, X_test, y_train, y_test

print("Data preprocessor created!")

# CELL 4: Model Definition
class ShelterPredictionModel:
    def __init__(self, model_type='lstm'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.canada_holidays = holidays.Canada()
        self.feature_names = None
        self.sequence_length = 7
        self.location_pipeline = LocationFeaturePipeline()
        
    def create_lstm_model(self, input_shape):
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
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        print(f"Training {self.model_type.upper()} model...")
        
        # GPU optimization: Increase batch size if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Training on GPU with {len(gpus)} device(s)")
            # Increase batch size for GPU training
            optimized_batch_size = min(batch_size * 2, 128)  # Cap at 128 to avoid memory issues
            print(f"Optimized batch size for GPU: {optimized_batch_size}")
        else:
            print("Training on CPU")
            optimized_batch_size = batch_size
        
        if self.model_type == 'lstm':
            self.model = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=optimized_batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=optimized_batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        print("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        
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
    
    def predict_for_shelter(self, shelter_info, target_date, historical_data=None):
        print(f"Predicting occupancy for shelter: {shelter_info.get('name', 'Unknown')}")
        print(f"Target_date: {target_date}")
        
        # Get location features
        location_features = self.location_pipeline.get_location_features_for_prediction(shelter_info)
        
        # Prepare features for the target date
        features = self.prepare_prediction_features(target_date, historical_data)
        
        # Add location features to the feature vector
        location_feature_vector = np.array([
            location_features.get('sector_encoded', 0),
            location_features.get('sector_avg_income', 0) / 100000,
            location_features.get('sector_population_density', 0) / 10000,
            location_features.get('sector_transit_accessibility', 0),
            location_features.get('sector_crime_rate', 0),
            location_features.get('sector_homelessness_rate', 0)
        ])
        
        # Combine temporal and location features
        combined_features = np.concatenate([features, location_feature_vector.reshape(1, -1)], axis=1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_reshaped = combined_features.reshape(-1, combined_features.shape[-1])
            features_scaled = self.scaler.transform(features_reshaped)
            combined_features = features_scaled.reshape(combined_features.shape)
        
        # Make prediction
        prediction = self.model.predict(combined_features.reshape(1, *combined_features.shape))
        predicted_occupancy = prediction[0][0]
        
        # Apply shelter-specific scaling if capacity is provided
        if 'maxCapacity' in shelter_info:
            max_capacity = shelter_info['maxCapacity']
            scaled_prediction = min(predicted_occupancy * (max_capacity / 100), max_capacity)
            predicted_occupancy = max(0, scaled_prediction)
        
        return {
            'shelter_name': shelter_info.get('name', 'Unknown'),
            'target_date': target_date,
            'predicted_occupancy': round(predicted_occupancy, 0),
            'max_capacity': shelter_info.get('maxCapacity', None),
            'utilization_rate': round((predicted_occupancy / shelter_info.get('maxCapacity', 100)) * 100, 1) if shelter_info.get('maxCapacity') else None,
            'sector_info': {
                'sector_id': location_features.get('sector_id', 'unknown'),
                'sector_name': location_features.get('sector_name', 'Unknown'),
                'sector_description': location_features.get('sector_description', 'Unknown')
            },
            'location_features': location_features
        }
    
    def prepare_prediction_features(self, target_date, historical_data=None):
        target_dt = pd.to_datetime(target_date)
        start_date = target_dt - timedelta(days=self.sequence_length)
        
        sequence_features = []
        
        for i in range(self.sequence_length):
            current_date = start_date + timedelta(days=i)
            
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
            
            if historical_data is not None:
                features['dow_avg'] = historical_data.get('dow_avg', {}).get(current_date.dayofweek, 0)
                features['month_avg'] = historical_data.get('month_avg', {}).get(current_date.month, 0)
            else:
                features['dow_avg'] = 0
                features['month_avg'] = 0
            
            sequence_features.append(list(features.values()))
        
        return np.array(sequence_features)
    
    def _get_season(self, month):
        if month in [12, 1, 2]: return 1  # Winter
        elif month in [3, 4, 5]: return 2  # Spring
        elif month in [6, 7, 8]: return 3  # Summer
        else: return 4  # Fall
    
    def _simulate_temperature(self, date):
        season = self._get_season(date.month)
        if season == 1: return np.random.normal(-5, 10)
        elif season == 2: return np.random.normal(10, 8)
        elif season == 3: return np.random.normal(25, 8)
        else: return np.random.normal(15, 8)
    
    def _simulate_precipitation(self, date):
        season = self._get_season(date.month)
        precip_prob = {1: 0.3, 2: 0.4, 3: 0.2, 4: 0.3}[season]
        if np.random.random() < precip_prob:
            return np.random.exponential(5)
        return 0
    
    def _get_weather_severity(self, date):
        temp = self._simulate_temperature(date)
        precip = self._simulate_precipitation(date)
        if (temp < -10) or (temp > 35) or (precip > 20): return 3
        elif (temp < 0) or (temp > 25) or (precip > 10): return 2
        else: return 1

print("Model class created!")

# CELL 5: GPU Monitoring Function
def monitor_gpu_usage():
    """Monitor GPU usage during training"""
    try:
        # Get GPU memory info
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Check if TensorFlow can see the GPU
            print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")
            
            # Test GPU computation
            with tf.device('/GPU:0'):
                test_tensor = tf.random.normal([1000, 1000])
                result = tf.matmul(test_tensor, test_tensor)
                print("✓ GPU computation test successful")
        else:
            print("No GPU devices found")
    except Exception as e:
        print(f"GPU monitoring error: {e}")

# CELL 6: Main Execution
def main():
    print("Shelter Prediction with Location Features")
    print("=" * 50)
    
    # GPU Status Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s) available")
        print(f"✓ TensorFlow will use GPU for training")
    else:
        print("⚠ No GPU detected - training will use CPU")
    
    # Step 1: Data Preprocessing
    print("\n1. Loading and preprocessing data...")
    preprocessor = ShelterDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(sequence_length=30)
    
    # Step 2: Model Training
    print("\n2. Training model...")
    model = ShelterPredictionModel(model_type='lstm')
    history = model.train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    
    # Step 3: Model Evaluation
    print("\n3. Evaluating model...")
    results = model.evaluate_model(X_test, y_test)
    
    # Step 4: Example Predictions
    print("\n4. Making example predictions...")
    
    shelter_info = {
        'name': 'Test Shelter',
        'maxCapacity': 100,
        'address': '100 Test Street',
        'postal_code': 'M5S 2P1',
        'city': 'Toronto',
        'province': 'ON'
    }
    
    tomorrow = datetime.now() + timedelta(days=1)
    target_date = tomorrow.strftime('%Y-%m-%d')
    
    prediction = model.predict_for_shelter(shelter_info, target_date)
    
    print(f"\nPrediction Results:")
    print(f"Shelter: {prediction['shelter_name']}")
    print(f"Date: {prediction['target_date']}")
    print(f"Predicted Occupancy: {prediction['predicted_occupancy']}")
    print(f"Max Capacity: {prediction['max_capacity']}")
    print(f"Utilization Rate: {prediction['utilization_rate']}%")
    print(f"Sector: {prediction['sector_info']['sector_name']}")
    print(f"Sector Description: {prediction['sector_info']['sector_description']}")
    
    # Step 5: Save Model
    print("\n5. Saving model...")
    os.makedirs('models', exist_ok=True)
    model.model.save('models/shelter_model_with_location.h5')
    joblib.dump(preprocessor.scaler, 'models/scaler_with_location.pkl')
    joblib.dump(preprocessor.location_pipeline, 'models/location_pipeline.pkl')
    
    print("Model and pipeline saved successfully!")
    
    return model, preprocessor, results

# CELL 7: GPU Monitoring Test
print("Testing GPU availability...")
monitor_gpu_usage()

# CELL 8: Run the Pipeline
model, preprocessor, results = main()

# CELL 9: Test with New Shelter
print("\n" + "="*50)
print("TESTING WITH NEW SHELTER")
print("="*50)

new_shelter_info = {
    'name': 'New Downtown Shelter',
    'maxCapacity': 150,
    'address': '200 Bay Street',
    'postal_code': 'M5J 2T3',
    'city': 'Toronto',
    'province': 'ON'
}

tomorrow = datetime.now() + timedelta(days=1)
target_date = tomorrow.strftime('%Y-%m-%d')

new_prediction = model.predict_for_shelter(new_shelter_info, target_date)

print(f"\nNew Shelter Prediction:")
print(f"Shelter: {new_prediction['shelter_name']}")
print(f"Date: {new_prediction['target_date']}")
print(f"Predicted Occupancy: {new_prediction['predicted_occupancy']}")
print(f"Max Capacity: {new_prediction['max_capacity']}")
print(f"Utilization Rate: {new_prediction['utilization_rate']}%")
print(f"Sector: {new_prediction['sector_info']['sector_name']}")
print(f"Sector Description: {new_prediction['sector_info']['sector_description']}")
print(f"Location Features: {new_prediction['location_features']}")

# CELL 10: Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Training History
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Actual vs Predicted
plt.subplot(1, 3, 2)
plt.scatter(results['actual'], results['predictions'], alpha=0.6)
plt.plot([results['actual'].min(), results['actual'].max()], 
         [results['actual'].min(), results['actual'].max()], 'r--', lw=2)
plt.xlabel('Actual Occupancy')
plt.ylabel('Predicted Occupancy')
plt.title('Actual vs Predicted')
plt.grid(True)

# Plot 3: Sector Distribution
plt.subplot(1, 3, 3)
sector_counts = preprocessor.location_pipeline.sector_mapper.sector_encoder.classes_
sector_data = [len(preprocessor.location_pipeline.sector_mapper.toronto_sectors[sector]['postal_codes']) 
               for sector in sector_counts]
plt.bar(sector_counts, sector_data)
plt.title('Postal Codes per Sector')
plt.xlabel('Sector')
plt.ylabel('Number of Postal Codes')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("Complete! The model has been trained with location features and can predict for new shelters.") 