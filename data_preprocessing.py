import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import holidays
import requests
import warnings
warnings.filterwarnings('ignore')

class ShelterDataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.shelter_names = None
        self.canada_holidays = holidays.Canada()
        
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

    def load_and_combine_data(self, file_paths):
        """Load and combine all CSV files"""
        print("Loading and combining data...")
        all_data = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                df['YEAR'] = pd.to_datetime(df['OCCUPANCY_DATE']).dt.year
                all_data.append(df)
                print(f"Loaded {len(df)} records from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not all_data:
            raise ValueError("No data files could be loaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} records")
        return combined_df
    
    def aggregate_across_shelters(self, df):
        """Step 1: Aggregate visitation across all shelters"""
        print("Aggregating data across all shelters...")
        
        # Convert date and create daily aggregates
        df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'])
        
        # Group by date and sum occupancy across all shelters
        daily_aggregate = df.groupby('OCCUPANCY_DATE').agg({
            'OCCUPANCY': 'sum',
            'SHELTER_NAME': 'nunique'  # Count unique shelters per day
        }).reset_index()
        
        daily_aggregate.columns = ['date', 'total_occupancy', 'shelter_count']
        daily_aggregate['avg_occupancy_per_shelter'] = daily_aggregate['total_occupancy'] / daily_aggregate['shelter_count']
        
        print(f"Aggregated data: {len(daily_aggregate)} days")
        print(f"Average occupancy per shelter: {daily_aggregate['avg_occupancy_per_shelter'].mean():.2f}")
        
        return daily_aggregate
    
    def add_temporal_features(self, df):
        """Add temporal features from date"""
        print("Adding temporal features...")
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['season'] = df['month'].map({
            12: 1, 1: 1, 2: 1,    # Winter
            3: 2, 4: 2, 5: 2,      # Spring
            6: 3, 7: 3, 8: 3,      # Summer
            9: 4, 10: 4, 11: 4     # Fall
        })
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Month end/beginning indicators
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        
        return df
    
    def add_holiday_features(self, df):
        """Add holiday features"""
        print("Adding holiday features...")
        
        # Canada holidays
        df['is_holiday'] = df['date'].apply(lambda x: x in self.canada_holidays).astype(int)
        df['holiday_name'] = df['date'].apply(lambda x: self.canada_holidays.get(x, ''))
        
        # Major holidays (binary indicators)
        major_holidays = [
            'New Year\'s Day', 'Canada Day', 'Labour Day', 
            'Thanksgiving Day', 'Christmas Day', 'Boxing Day'
        ]
        
        for holiday in major_holidays:
            df[f'is_{holiday.lower().replace(" ", "_").replace("\'", "")}'] = (
                df['holiday_name'].str.contains(holiday, case=False, na=False)
            ).astype(int)
        
        return df
    
    def add_weather_features(self, df):
        """Add weather features (simulated for now)"""
        print("Adding weather features...")
        
        # Simulate weather data based on season and month
        np.random.seed(42)  # For reproducibility
        
        # Temperature (simulated based on season)
        df['temperature'] = df['season'].map({
            1: lambda: np.random.normal(-5, 10),   # Winter
            2: lambda: np.random.normal(10, 8),    # Spring
            3: lambda: np.random.normal(25, 8),    # Summer
            4: lambda: np.random.normal(15, 8)     # Fall
        }).apply(lambda x: x())
        
        # Precipitation probability
        df['precipitation_prob'] = df['season'].map({
            1: 0.3,  # Winter - snow
            2: 0.4,  # Spring - rain
            3: 0.2,  # Summer - occasional rain
            4: 0.3   # Fall - rain
        })
        
        # Precipitation amount (mm)
        df['precipitation_mm'] = np.where(
            np.random.random(len(df)) < df['precipitation_prob'],
            np.random.exponential(5, len(df)),
            0
        )
        
        # Weather severity (1=mild, 2=moderate, 3=severe)
        df['weather_severity'] = np.where(
            (df['temperature'] < -10) | (df['temperature'] > 35) | (df['precipitation_mm'] > 20),
            3,
            np.where(
                (df['temperature'] < 0) | (df['temperature'] > 25) | (df['precipitation_mm'] > 10),
                2,
                1
            )
        )
        
        return df
    
    def add_trend_features(self, df):
        """Add trend and lag features"""
        print("Adding trend features...")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Lag features (previous days)
        for lag in [1, 2, 3, 7, 14]:
            df[f'occupancy_lag_{lag}'] = df['avg_occupancy_per_shelter'].shift(lag)
        
        # Rolling averages
        for window in [3, 7, 14, 30]:
            df[f'occupancy_rolling_mean_{window}'] = df['avg_occupancy_per_shelter'].rolling(window=window, min_periods=1).mean()
            df[f'occupancy_rolling_std_{window}'] = df['avg_occupancy_per_shelter'].rolling(window=window, min_periods=1).std()
        
        # Trend indicators
        df['occupancy_trend_7d'] = df['avg_occupancy_per_shelter'] - df['occupancy_lag_7']
        df['occupancy_trend_14d'] = df['avg_occupancy_per_shelter'] - df['occupancy_lag_14']
        
        # Day of week averages
        dow_avg = df.groupby('day_of_week')['avg_occupancy_per_shelter'].mean()
        df['dow_avg'] = df['day_of_week'].map(dow_avg)
        
        # Month averages
        month_avg = df.groupby('month')['avg_occupancy_per_shelter'].mean()
        df['month_avg'] = df['month'].map(month_avg)
        
        return df
    
    def create_sequences_for_prediction(self, df, sequence_length=7):
        """Create sequences for time series prediction"""
        print(f"Creating sequences with length {sequence_length}...")
        
        features = [
            'avg_occupancy_per_shelter', 'total_occupancy', 'shelter_count',
            'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year', 'quarter',
            'season', 'is_weekend', 'is_month_end', 'is_month_start',
            'is_holiday', 'temperature', 'precipitation_mm', 'weather_severity',
            'occupancy_lag_1', 'occupancy_lag_2', 'occupancy_lag_3', 'occupancy_lag_7',
            'occupancy_rolling_mean_7', 'occupancy_rolling_std_7',
            'occupancy_trend_7d', 'dow_avg', 'month_avg'
        ]
        
        # Remove rows with NaN values
        df_clean = df[features].dropna()
        
        sequences = []
        targets = []
        
        for i in range(len(df_clean) - sequence_length):
            seq = df_clean.iloc[i:i+sequence_length].values
            target = df_clean.iloc[i+sequence_length]['avg_occupancy_per_shelter']
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def scale_features_for_prediction(self, X_train, X_test=None):
        """Scale features"""
        print("Scaling features...")
        
        # Reshape for scaling (flatten sequences)
        original_shape = X_train.shape
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(original_shape)
        
        if X_test is not None:
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_prediction_features(self, target_date, days_ahead=7):
        """Prepare features for prediction at inference time"""
        print(f"Preparing prediction features for {target_date}...")
        
        # Create date range
        start_date = pd.to_datetime(target_date) - timedelta(days=30)
        end_date = pd.to_datetime(target_date) + timedelta(days=days_ahead)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create base dataframe
        pred_df = pd.DataFrame({'date': date_range})
        
        # Add all features
        pred_df = self.add_temporal_features(pred_df)
        pred_df = self.add_holiday_features(pred_df)
        pred_df = self.add_weather_features(pred_df)
        
        # For prediction, we'll need to fill in some lag features
        # This would typically come from recent historical data
        # For now, we'll use averages
        
        features = [
            'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year', 'quarter',
            'season', 'is_weekend', 'is_month_end', 'is_month_start',
            'is_holiday', 'temperature', 'precipitation_mm', 'weather_severity',
            'dow_avg', 'month_avg'
        ]
        
        return pred_df[features]
    
    def process_data(self, file_paths, test_size=0.2, sequence_length=7):
        """Main processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Step 1: Load and combine data
        raw_data = self.load_and_combine_data(file_paths)
        
        # Step 2: Aggregate across shelters
        daily_data = self.aggregate_across_shelters(raw_data)
        
        # Step 3: Add universal features
        daily_data = self.add_temporal_features(daily_data)
        daily_data = self.add_holiday_features(daily_data)
        daily_data = self.add_weather_features(daily_data)
        daily_data = self.add_trend_features(daily_data)
        
        # Step 4: Create sequences
        X, y = self.create_sequences_for_prediction(daily_data, sequence_length)
        
        # Step 5: Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Step 6: Scale features
        X_train_scaled, X_test_scaled = self.scale_features_for_prediction(X_train, X_test)
        
        print(f"Final dataset shapes:")
        print(f"X_train: {X_train_scaled.shape}")
        print(f"X_test: {X_test_scaled.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'daily_data': daily_data,
            'scaler': self.scaler
        }

if __name__ == "__main__":
    preprocessor = ShelterDataPreprocessor()
    
    # File paths
    file_paths = [
        'data/Daily shelter occupancy 2017.csv',
        'data/Daily shelter occupancy 2018.csv',
        'data/Daily shelter occupancy 2019.csv',
        'data/Daily shelter occupancy 2020.csv'
    ]
    
    # Process data
    processed_data = preprocessor.process_data(file_paths)
    
    print("\nData processing completed!")
    print(f"Training samples: {len(processed_data['X_train'])}")
    print(f"Test samples: {len(processed_data['X_test'])}")
    
    preprocessor.save_preprocessors()
    
    print("Data preprocessing completed!")
    print(f"Training set: {processed_data['X_train'].shape}")
    print(f"Test set: {processed_data['X_test'].shape}") 