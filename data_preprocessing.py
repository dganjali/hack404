import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class ShelterDataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.shelter_names = None
        
    def load_data(self):
        """Load all CSV files and combine them"""
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
        """Preprocess the data for training"""
        # Convert date column
        df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'], errors='coerce', infer_datetime_format=True)
        # Drop rows with unparseable dates
        df = df.dropna(subset=['OCCUPANCY_DATE'])
        
        # Extract date features
        df['year'] = df['OCCUPANCY_DATE'].dt.year
        df['month'] = df['OCCUPANCY_DATE'].dt.month
        df['day'] = df['OCCUPANCY_DATE'].dt.day
        df['day_of_week'] = df['OCCUPANCY_DATE'].dt.dayofweek
        df['day_of_year'] = df['OCCUPANCY_DATE'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclical features for time
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Encode categorical variables
        categorical_cols = ['ORGANIZATION_NAME', 'SHELTER_NAME', 'SECTOR', 'PROGRAM_NAME']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Store shelter names for later use
        self.shelter_names = df['SHELTER_NAME'].unique()
        
        return df
    
    def create_sequences(self, df, sequence_length=30):
        """Create time series sequences for training"""
        sequences = []
        targets = []
        
        # Group by shelter and date
        df_sorted = df.sort_values(['SHELTER_NAME', 'OCCUPANCY_DATE'])
        
        for shelter in df['SHELTER_NAME'].unique():
            shelter_data = df_sorted[df_sorted['SHELTER_NAME'] == shelter].copy()
            
            if len(shelter_data) < sequence_length + 1:
                continue
                
            # Create features for sequence
            feature_cols = [
                'year', 'month', 'day', 'day_of_week', 'day_of_year', 'is_weekend',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 
                'day_of_week_sin', 'day_of_week_cos',
                'ORGANIZATION_NAME_encoded', 'SHELTER_NAME_encoded', 
                'SECTOR_encoded', 'PROGRAM_NAME_encoded'
            ]
            
            # Add lagged occupancy features
            for lag in range(1, 8):  # 1-7 day lags
                shelter_data[f'occupancy_lag_{lag}'] = shelter_data['OCCUPANCY'].shift(lag)
            
            # Fill NaN values with 0 for lag features
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
        """Main method to prepare data for training"""
        print("Loading data...")
        df = self.load_data()
        
        print("Preprocessing data...")
        df = self.preprocess_data(df)
        
        print("Creating sequences...")
        X, y = self.create_sequences(df, sequence_length)
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Dynamically determine n_features
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
    
    def save_preprocessors(self, save_dir='models'):
        """Save preprocessors for later use"""
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.shelter_names, os.path.join(save_dir, 'shelter_names.pkl'))
        
        print(f"Preprocessors saved to {save_dir}/")
    
    def load_preprocessors(self, save_dir='models'):
        """Load saved preprocessors"""
        self.label_encoders = joblib.load(os.path.join(save_dir, 'label_encoders.pkl'))
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        self.shelter_names = joblib.load(os.path.join(save_dir, 'shelter_names.pkl'))

if __name__ == "__main__":
    preprocessor = ShelterDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    preprocessor.save_preprocessors()
    
    print("Data preprocessing completed!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}") 