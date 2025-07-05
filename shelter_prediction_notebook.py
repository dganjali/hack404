# Shelter Prediction with Location Features - Complete Jupyter Notebook
# This notebook implements the sector-based location feature pipeline for shelter occupancy prediction

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

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

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

# ============================================================================
# 1. LOCATION FEATURES PIPELINE
# ============================================================================

class TorontoSectorMapper:
    """Toronto Sector Mapper for Shelter Location Features"""
    
    def __init__(self):
        # Define Toronto sectors based on postal code areas (FSA - Forward Sortation Areas)
        self.toronto_sectors = {
            # Downtown Core
            'downtown_core': {
                'name': 'Downtown Core',
                'postal_codes': ['M5A', 'M5B', 'M5C', 'M5E', 'M5G', 'M5H', 'M5J', 'M5K', 'M5L', 'M5M', 'M5N', 'M5P', 'M5R', 'M5S', 'M5T', 'M5V', 'M5W', 'M5X', 'M5Y', 'M5Z'],
                'description': 'Financial district, entertainment district, university area'
            },
            # East End
            'east_end': {
                'name': 'East End',
                'postal_codes': ['M1B', 'M1C', 'M1E', 'M1G', 'M1H', 'M1J', 'M1K', 'M1L', 'M1M', 'M1N', 'M1P', 'M1R', 'M1S', 'M1T', 'M1V', 'M1W', 'M1X'],
                'description': 'Scarborough, East York, Beaches'
            },
            # West End
            'west_end': {
                'name': 'West End',
                'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S'],
                'description': 'West Toronto, Parkdale, High Park, Junction'
            },
            # North End
            'north_end': {
                'name': 'North End',
                'postal_codes': ['M2H', 'M2J', 'M2K', 'M2L', 'M2M', 'M2N', 'M2P', 'M2R', 'M3A', 'M3B', 'M3C', 'M3H', 'M3J', 'M3K', 'M3L', 'M3M', 'M3N', 'M4A', 'M4B', 'M4C', 'M4E', 'M4G', 'M4H', 'M4J', 'M4K', 'M4L', 'M4M', 'M4N', 'M4P', 'M4R', 'M4S', 'M4T', 'M4V', 'M4W', 'M4X', 'M4Y'],
                'description': 'North York, York, Don Mills, Lawrence Park'
            },
            # Etobicoke
            'etobicoke': {
                'name': 'Etobicoke',
                'postal_codes': ['M8V', 'M8W', 'M8X', 'M8Y', 'M8Z', 'M9A', 'M9B', 'M9C', 'M9P', 'M9R', 'M9V', 'M9W'],
                'description': 'Etobicoke, Rexdale, Humber Bay'
            },
            # York
            'york': {
                'name': 'York',
                'postal_codes': ['M6A', 'M6B', 'M6C', 'M6D', 'M6E', 'M6F', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L', 'M6M', 'M6N', 'M6P', 'M6R', 'M6S'],
                'description': 'York, Weston, Mount Dennis'
            }
        }
        
        # Sector socioeconomic indicators (simulated data - in practice, get from census)
        self.sector_indicators = {
            'downtown_core': {
                'avg_income': 85000,
                'population_density': 8500,
                'transit_accessibility': 0.95,
                'crime_rate': 0.3,
                'homelessness_rate': 0.8
            },
            'east_end': {
                'avg_income': 65000,
                'population_density': 4200,
                'transit_accessibility': 0.75,
                'crime_rate': 0.4,
                'homelessness_rate': 0.6
            },
            'west_end': {
                'avg_income': 72000,
                'population_density': 5800,
                'transit_accessibility': 0.85,
                'crime_rate': 0.35,
                'homelessness_rate': 0.7
            },
            'north_end': {
                'avg_income': 95000,
                'population_density': 3200,
                'transit_accessibility': 0.65,
                'crime_rate': 0.2,
                'homelessness_rate': 0.3
            },
            'etobicoke': {
                'avg_income': 78000,
                'population_density': 2800,
                'transit_accessibility': 0.55,
                'crime_rate': 0.25,
                'homelessness_rate': 0.4
            },
            'york': {
                'avg_income': 68000,
                'population_density': 3800,
                'transit_accessibility': 0.70,
                'crime_rate': 0.45,
                'homelessness_rate': 0.5
            }
        }
        
        # Initialize sector encoder with all known sectors plus 'unknown'
        all_sectors = list(self.toronto_sectors.keys()) + ['unknown']
        self.sector_encoder = LabelEncoder()
        self.sector_encoder.fit(all_sectors)
        
    def get_sector_from_postal_code(self, postal_code):
        """Map postal code to sector"""
        if pd.isna(postal_code) or postal_code == '':
            return 'unknown'
        
        # Extract first 3 characters (FSA)
        fsa = str(postal_code).strip()[:3].upper()
        
        for sector_id, sector_info in self.toronto_sectors.items():
            if fsa in sector_info['postal_codes']:
                return sector_id
        
        return 'unknown'
    
    def assign_shelter_to_sector(self, shelter_data):
        """Assign each shelter to a sector"""
        print("Assigning shelters to sectors...")
        
        df = shelter_data.copy()
        
        # Add sector column
        df['sector'] = df.apply(
            lambda row: self.get_sector_from_postal_code(row['SHELTER_POSTAL_CODE']) 
            if pd.notna(row['SHELTER_POSTAL_CODE']) 
            else 'unknown',
            axis=1
        )
        
        # Handle any sectors that might not be in the encoder
        unique_sectors = df['sector'].unique()
        for sector in unique_sectors:
            if sector not in self.sector_encoder.classes_:
                print(f"Warning: Unknown sector '{sector}' found, mapping to 'unknown'")
                df.loc[df['sector'] == sector, 'sector'] = 'unknown'
        
        # Encode sectors
        df['sector_encoded'] = self.sector_encoder.transform(df['sector'])
        
        # Add sector socioeconomic indicators
        for indicator in ['avg_income', 'population_density', 'transit_accessibility', 'crime_rate', 'homelessness_rate']:
            df[f'sector_{indicator}'] = df['sector'].map(
                {sector: self.sector_indicators[sector][indicator] 
                 for sector in self.sector_indicators.keys()}
            ).fillna(0)
        
        return df
    
    def create_sector_features(self, df):
        """Create sector-based features for prediction"""
        print("Creating sector-based features...")
        
        # One-hot encode sectors
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)
        
        # Create sector-level aggregates
        sector_stats = df.groupby('sector').agg({
            'OCCUPANCY': ['mean', 'std', 'min', 'max'],
            'SHELTER_NAME': 'nunique'  # Number of shelters per sector
        }).reset_index()
        
        sector_stats.columns = ['sector', 'sector_avg_occupancy', 'sector_std_occupancy', 
                              'sector_min_occupancy', 'sector_max_occupancy', 'shelters_in_sector']
        
        # Merge sector statistics back to main dataframe
        df = df.merge(sector_stats, on='sector', how='left')
        
        # Create sector trend features
        df['sector_occupancy_ratio'] = df['OCCUPANCY'] / df['sector_avg_occupancy']
        df['sector_occupancy_zscore'] = (df['OCCUPANCY'] - df['sector_avg_occupancy']) / df['sector_std_occupancy']
        
        return df
    
    def get_sector_for_new_shelter(self, address, postal_code=None, city='Toronto', province='ON'):
        """Get sector information for a new shelter"""
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
    """Complete location feature pipeline for shelter prediction"""
    
    def __init__(self):
        self.sector_mapper = TorontoSectorMapper()
        self.feature_columns = []
        
    def process_location_features(self, df):
        """Process location features for the entire dataset"""
        print("Processing location features...")
        
        # Step 1: Assign shelters to sectors
        df = self.sector_mapper.assign_shelter_to_sector(df)
        
        # Step 2: Create sector-based features
        df = self.sector_mapper.create_sector_features(df)
        
        # Store feature column names
        self.feature_columns = [col for col in df.columns if any(prefix in col for prefix in 
                                                               ['sector_', 'shelters_in_sector', 'sector_occupancy'])]
        
        print(f"Created {len(self.feature_columns)} location features")
        return df
    
    def get_location_features_for_prediction(self, shelter_info):
        """Get location features for a new shelter prediction"""
        sector_info = self.sector_mapper.get_sector_for_new_shelter(
            address=shelter_info.get('address', ''),
            postal_code=shelter_info.get('postal_code'),
            city=shelter_info.get('city', 'Toronto'),
            province=shelter_info.get('province', 'ON')
        )
        
        # Create feature vector
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

# ============================================================================
# 1.5. WEATHER DATA PROCESSOR
# ============================================================================

class WeatherDataProcessor:
    """Weather data processor - SIMPLIFIED FOR TEMPERATURE ONLY"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.weather_data = None
        self.daily_weather = None
        
    def load_weather_data(self):
        """Load all weather CSV files from all years - TEMPERATURE ONLY"""
        print("Loading weather data from year/month folders...")
        all_weather_data = []
        
        for year in [2017, 2018, 2019, 2020]:
            weather_dir = os.path.join(self.data_dir, f'Weather {year}')
            if os.path.exists(weather_dir):
                print(f"Loading weather data for {year}...")
                
                # Load all monthly files for this year
                for month in range(1, 13):
                    month_str = f"{month:02d}"
                    filename = f"en_climate_hourly_ON_6158359_{month_str}-{year}_P1H.csv"
                    file_path = os.path.join(weather_dir, filename)
                    
                    if os.path.exists(file_path):
                        try:
                            print(f"  Loading {filename}...")
                            df = pd.read_csv(file_path)
                            # Only keep essential columns: Date/Time, Temp (°C), Year, Month, Day
                            essential_cols = ['Date/Time', 'Temp (°C)', 'Year', 'Month', 'Day']
                            available_cols = [col for col in essential_cols if col in df.columns]
                            
                            if len(available_cols) >= 3:  # At least Date/Time, Temp, and one date component
                                df = df[available_cols]
                                all_weather_data.append(df)
                            else:
                                print(f"  Skipping {filename} - missing essential columns")
                        except Exception as e:
                            print(f"Error loading {filename}: {e}")
                    else:
                        print(f"  File not found: {filename}")
        
        if all_weather_data:
            self.weather_data = pd.concat(all_weather_data, ignore_index=True)
            print(f"Weather data loaded: {self.weather_data.shape}")
            print(f"Columns: {list(self.weather_data.columns)}")
        else:
            print("No weather data found!")
            return None
        
        return self.weather_data
    
    def process_weather_data(self):
        """Process weather data to get daily temperature averages - TEMPERATURE ONLY"""
        if self.weather_data is None:
            self.load_weather_data()
        
        if self.weather_data is None:
            return None
        
        print("Processing weather data for temperature only...")
        
        # Convert date/time column
        self.weather_data['Date/Time'] = pd.to_datetime(self.weather_data['Date/Time'])
        
        # Extract date components
        self.weather_data['date'] = self.weather_data['Date/Time'].dt.date
        self.weather_data['year'] = self.weather_data['Date/Time'].dt.year
        self.weather_data['month'] = self.weather_data['Date/Time'].dt.month
        self.weather_data['day'] = self.weather_data['Date/Time'].dt.day
        
        # Convert temperature to numeric, handling any non-numeric values
        self.weather_data['Temp (°C)'] = pd.to_numeric(self.weather_data['Temp (°C)'], errors='coerce')
        
        print(f"Processing weather data with columns: {list(self.weather_data.columns)}")
        
        # Calculate daily temperature averages - ONLY TEMPERATURE
        groupby_columns = ['date', 'year', 'month', 'day']
        
        # Verify all groupby columns exist
        missing_columns = [col for col in groupby_columns if col not in self.weather_data.columns]
        if missing_columns:
            print(f"Warning: Missing columns for groupby: {missing_columns}")
            print(f"Available columns: {list(self.weather_data.columns)}")
            # Remove missing columns from groupby
            groupby_columns = [col for col in groupby_columns if col in self.weather_data.columns]
            print(f"Using groupby columns: {groupby_columns}")
        
        # Only aggregate temperature - nothing else
        daily_weather = self.weather_data.groupby(groupby_columns).agg({
            'Temp (°C)': 'mean'  # ONLY TEMPERATURE MEAN
        }).reset_index()
        
        # Rename temperature column to be consistent
        daily_weather = daily_weather.rename(columns={'Temp (°C)': 'temp_mean'})
        
        print(f"Daily weather columns: {list(daily_weather.columns)}")
        print(f"Sample data shape: {daily_weather.shape}")
        
        self.daily_weather = daily_weather
        print(f"Processed daily weather data: {self.daily_weather.shape}")
        
        return self.daily_weather
    
    def get_weather_for_date(self, target_date):
        """Get temperature data for a specific date - TEMPERATURE ONLY"""
        if self.daily_weather is None:
            self.process_weather_data()
        
        if self.daily_weather is None:
            return None
        
        target_dt = pd.to_datetime(target_date)
        target_date_only = target_dt.date()
        
        # Find weather data for this date
        weather_row = self.daily_weather[self.daily_weather['date'] == target_date_only]
        
        if len(weather_row) == 0:
            # If no exact match, use monthly average
            month = target_dt.month
            
            try:
                monthly_avg = self.daily_weather[self.daily_weather['month'] == month]['temp_mean'].mean()
                if pd.isna(monthly_avg):
                    monthly_avg = self._simulate_temp_by_month(month)
            except:
                monthly_avg = self._simulate_temp_by_month(month)
            
            return {
                'temp_mean': monthly_avg,
                'month': month,
                'day': target_dt.day
            }
        
        # Return actual temperature data
        row = weather_row.iloc[0]
        
        return {
            'temp_mean': row['temp_mean'],
            'month': row['month'],
            'day': row['day']
        }
    
    def _simulate_temp_by_month(self, month):
        """Simulate temperature based on month"""
        if month in [12, 1, 2]:  # Winter
            return np.random.normal(-5, 10)
        elif month in [3, 4, 5]:  # Spring
            return np.random.normal(10, 8)
        elif month in [6, 7, 8]:  # Summer
            return np.random.normal(25, 8)
        else:  # Fall
            return np.random.normal(15, 8)

# ============================================================================
# 2. DATA PREPROCESSING - YEAR-INDEPENDENT WITH WEATHER
# ============================================================================

class ShelterDataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.shelter_names = None
        self.canada_holidays = holidays.Canada()
        self.location_pipeline = LocationFeaturePipeline()
        self.weather_processor = WeatherDataProcessor(data_dir)
        
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
        """Preprocess the data for training - YEAR INDEPENDENT WITH WEATHER"""
        # Convert date column
        df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'], errors='coerce', infer_datetime_format=True)
        df = df.dropna(subset=['OCCUPANCY_DATE'])
        
        # Extract date features - IGNORE YEAR, focus on day/month patterns
        df['month'] = df['OCCUPANCY_DATE'].dt.month
        df['day'] = df['OCCUPANCY_DATE'].dt.day
        df['day_of_week'] = df['OCCUPANCY_DATE'].dt.dayofweek
        df['day_of_year'] = df['OCCUPANCY_DATE'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclical features for time - YEAR INDEPENDENT
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Add seasonal features
        df['season'] = df['month'].apply(self._get_season)
        df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
        df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
        
        # Add holiday features
        df['is_holiday'] = df['OCCUPANCY_DATE'].apply(lambda x: x in self.canada_holidays).astype(int)
        
        # Process weather data - TEMPERATURE ONLY
        print("Processing weather data for temperature only...")
        weather_data = self.weather_processor.process_weather_data()
        
        if weather_data is not None:
            # Merge weather data with shelter data
            # Convert both to string format for consistent merging
            df['date_str'] = df['OCCUPANCY_DATE'].dt.strftime('%Y-%m-%d')
            weather_data['date_str'] = weather_data['date'].astype(str)
            
            # Merge on date string
            df = df.merge(weather_data[['date_str', 'temp_mean', 'month', 'day']], 
                         on='date_str', 
                         how='left', 
                         suffixes=('', '_weather'))
            
            # Fill missing temperature data with monthly averages
            if 'temp_mean' in df.columns:
                monthly_avg = df.groupby('month')['temp_mean'].mean()
                df['temp_mean'] = df['temp_mean'].fillna(df['month'].map(monthly_avg))
            
            print(f"Added temperature feature")
        else:
            print("Warning: No weather data available, using simulated temperature")
            # Add simulated temperature
            df['temp_mean'] = df['month'].apply(lambda x: self._simulate_temperature_by_month(x))
        
        # Process location features using sector method
        df = self.location_pipeline.process_location_features(df)
        
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
    
    def _simulate_temperature_by_month(self, month):
        """Simulate temperature based on month"""
        if month in [12, 1, 2]:  # Winter
            return np.random.normal(-5, 10)
        elif month in [3, 4, 5]:  # Spring
            return np.random.normal(10, 8)
        elif month in [6, 7, 8]:  # Summer
            return np.random.normal(25, 8)
        else:  # Fall
            return np.random.normal(15, 8)
    

    
    def create_sequences(self, df, sequence_length=30):
        """Create time series sequences for training - YEAR INDEPENDENT WITH WEATHER"""
        sequences = []
        targets = []
        
        # Group by shelter and date
        df_sorted = df.sort_values(['SHELTER_NAME', 'OCCUPANCY_DATE'])
        
        for shelter in df['SHELTER_NAME'].unique():
            shelter_data = df_sorted[df_sorted['SHELTER_NAME'] == shelter].copy()
            
            if len(shelter_data) < sequence_length + 1:
                continue
                
            # Create features for sequence - YEAR INDEPENDENT WITH WEATHER
            feature_cols = [
                'month', 'day', 'day_of_week', 'day_of_year', 'is_weekend',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 
                'day_of_week_sin', 'day_of_week_cos',
                'season', 'season_sin', 'season_cos', 'is_holiday',
                'ORGANIZATION_NAME_encoded', 'SHELTER_NAME_encoded', 
                'SECTOR_encoded', 'PROGRAM_NAME_encoded'
            ]
            
            # Add temperature feature only
            if 'temp_mean' in shelter_data.columns:
                feature_cols.append('temp_mean')
            
            # Add location features
            location_features = [col for col in shelter_data.columns if any(prefix in col for prefix in 
                                                                          ['sector_', 'shelters_in_sector', 'sector_occupancy'])]
            feature_cols.extend(location_features)
            
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
        """Main method to prepare data for training - YEAR INDEPENDENT WITH WEATHER"""
        print("Loading data...")
        df = self.load_data()
        
        print("Preprocessing data (year-independent with weather)...")
        df = self.preprocess_data(df)
        
        print("Creating sequences...")
        X, y = self.create_sequences(df, sequence_length)
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
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

# ============================================================================
# 3. MODEL DEFINITION
# ============================================================================

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
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the model"""
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
        
        # Create model based on type
        if self.model_type == 'lstm':
            self.model = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
        elif self.model_type == 'conv_lstm':
            self.model = self.create_conv_lstm_model((X_train.shape[1], X_train.shape[2]))
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
    
    def predict_for_shelter(self, shelter_info, target_date, historical_data=None):
        """Predict occupancy for a specific shelter on a specific date - YEAR INDEPENDENT"""
        print(f"Predicting occupancy for shelter: {shelter_info.get('name', 'Unknown')}")
        print(f"Target_date: {target_date}")
        
        # Get location features for the shelter
        location_features = self.location_pipeline.get_location_features_for_prediction(shelter_info)
        
        # Prepare features for the target date - YEAR INDEPENDENT
        features = self.prepare_prediction_features(target_date, historical_data)
        
        # Add location features to the feature vector
        location_feature_vector = np.array([
            location_features.get('sector_encoded', 0),
            location_features.get('sector_avg_income', 0) / 100000,  # Normalize
            location_features.get('sector_population_density', 0) / 10000,  # Normalize
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
        """Prepare features for a specific date prediction - YEAR INDEPENDENT WITH WEATHER"""
        print(f"Preparing features for {target_date}...")
        
        # Create date range for sequence
        target_dt = pd.to_datetime(target_date)
        start_date = target_dt - timedelta(days=self.sequence_length)
        
        # Generate features for the sequence - YEAR INDEPENDENT WITH WEATHER
        sequence_features = []
        
        for i in range(self.sequence_length):
            current_date = start_date + timedelta(days=i)
            
            # Temporal features - YEAR INDEPENDENT
            features = {
                'month': current_date.month,
                'day': current_date.day,
                'day_of_week': current_date.dayofweek,
                'day_of_year': current_date.dayofyear,
                'is_weekend': int(current_date.dayofweek >= 5),
                'month_sin': np.sin(2 * np.pi * current_date.month / 12),
                'month_cos': np.cos(2 * np.pi * current_date.month / 12),
                'day_sin': np.sin(2 * np.pi * current_date.day / 31),
                'day_cos': np.cos(2 * np.pi * current_date.day / 31),
                'day_of_week_sin': np.sin(2 * np.pi * current_date.dayofweek / 7),
                'day_of_week_cos': np.cos(2 * np.pi * current_date.dayofweek / 7),
                'season': self._get_season(current_date.month),
                'season_sin': np.sin(2 * np.pi * self._get_season(current_date.month) / 4),
                'season_cos': np.cos(2 * np.pi * self._get_season(current_date.month) / 4),
                'is_holiday': int(current_date in self.canada_holidays)
            }
            
            # Add temperature feature only
            weather_info = self._get_weather_for_date(current_date)
            if weather_info:
                features.update({
                    'temp_mean': weather_info.get('temp_mean', 10)
                })
            else:
                # Fallback to simulated temperature
                features.update({
                    'temp_mean': self._simulate_temperature(current_date)
                })
            
            # Add historical averages if available
            if historical_data is not None:
                features['dow_avg'] = historical_data.get('dow_avg', {}).get(current_date.dayofweek, 0)
                features['month_avg'] = historical_data.get('month_avg', {}).get(current_date.month, 0)
            else:
                features['dow_avg'] = 0
                features['month_avg'] = 0
            
            sequence_features.append(list(features.values()))
        
        return np.array(sequence_features)
    
    def _get_weather_for_date(self, date):
        """Get temperature data for a specific date - TEMPERATURE ONLY"""
        # This would be implemented to use the WeatherDataProcessor
        # For now, return simulated temperature
        return {
            'temp_mean': self._simulate_temperature(date),
            'month': date.month,
            'day': date.day
        }
    
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
    


# ============================================================================
# 4. MAIN EXECUTION SCRIPT
# ============================================================================

def main():
    """Main execution function"""
    print("Shelter Prediction with Location Features - YEAR INDEPENDENT WITH WEATHER")
    print("=" * 70)
    
    # GPU Status Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s) available")
        print(f"✓ TensorFlow will use GPU for training")
    else:
        print("⚠ No GPU detected - training will use CPU")
    
    # Step 1: Data Preprocessing
    print("\n1. Loading and preprocessing data (year-independent with weather)...")
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
    
    # Example shelter info
    shelter_info = {
        'name': 'Test Shelter',
        'maxCapacity': 100,
        'address': '100 Test Street',
        'postal_code': 'M5S 2P1',
        'city': 'Toronto',
        'province': 'ON'
    }
    
    # Predict for tomorrow
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
    model.model.save('models/shelter_model_with_weather.h5')
    joblib.dump(preprocessor.scaler, 'models/scaler_with_weather.pkl')
    joblib.dump(preprocessor.location_pipeline, 'models/location_pipeline_with_weather.pkl')
    joblib.dump(preprocessor.weather_processor, 'models/weather_processor.pkl')
    
    print("Model and pipeline saved successfully!")
    
    return model, preprocessor, results

# ============================================================================
# 5. PREDICTION FUNCTION FOR NEW SHELTERS
# ============================================================================

def predict_for_new_shelter(shelter_info, target_date, model_path='models/shelter_model_with_weather.h5'):
    """Predict occupancy for a new shelter using location features - YEAR INDEPENDENT WITH WEATHER"""
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create prediction model instance
    predictor = ShelterPredictionModel()
    predictor.model = model
    
    # Make prediction
    prediction = predictor.predict_for_shelter(shelter_info, target_date)
    
    return prediction

# ============================================================================
# 6. USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Test GPU availability first
    print("Testing GPU availability...")
    monitor_gpu_usage()
    
    # Run the complete pipeline
    model, preprocessor, results = main()
    
    # Example: Predict for a new shelter in 2025 with weather
    print("\n" + "="*60)
    print("EXAMPLE: Predicting for a new shelter in 2025 with weather data")
    print("="*60)
    
    new_shelter_info = {
        'name': 'New Downtown Shelter',
        'maxCapacity': 150,
        'address': '200 Bay Street',
        'postal_code': 'M5J 2T3',
        'city': 'Toronto',
        'province': 'ON'
    }
    
    # Test prediction for 2025 winter
    target_date_2025_winter = '2025-01-15'  # January 15, 2025
    
    winter_prediction = predict_for_new_shelter(new_shelter_info, target_date_2025_winter)
    
    print(f"\n2025 Winter Prediction Results:")
    print(f"Shelter: {winter_prediction['shelter_name']}")
    print(f"Date: {winter_prediction['target_date']}")
    print(f"Predicted Occupancy: {winter_prediction['predicted_occupancy']}")
    print(f"Max Capacity: {winter_prediction['max_capacity']}")
    print(f"Utilization Rate: {winter_prediction['utilization_rate']}%")
    print(f"Sector: {winter_prediction['sector_info']['sector_name']}")
    print(f"Sector Description: {winter_prediction['sector_info']['sector_description']}")
    
    # Test prediction for 2025 summer
    target_date_2025_summer = '2025-07-15'  # July 15, 2025
    
    summer_prediction = predict_for_new_shelter(new_shelter_info, target_date_2025_summer)
    
    print(f"\n2025 Summer Prediction Results:")
    print(f"Shelter: {summer_prediction['shelter_name']}")
    print(f"Date: {summer_prediction['target_date']}")
    print(f"Predicted Occupancy: {summer_prediction['predicted_occupancy']}")
    print(f"Max Capacity: {summer_prediction['max_capacity']}")
    print(f"Utilization Rate: {summer_prediction['utilization_rate']}%")
    print(f"Sector: {summer_prediction['sector_info']['sector_name']}")
    print(f"Sector Description: {summer_prediction['sector_info']['sector_description']}")
    
    # Test prediction for 2025 fall
    target_date_2025_fall = '2025-10-15'  # October 15, 2025
    
    fall_prediction = predict_for_new_shelter(new_shelter_info, target_date_2025_fall)
    
    print(f"\n2025 Fall Prediction Results:")
    print(f"Shelter: {fall_prediction['shelter_name']}")
    print(f"Date: {fall_prediction['target_date']}")
    print(f"Predicted Occupancy: {fall_prediction['predicted_occupancy']}")
    print(f"Max Capacity: {fall_prediction['max_capacity']}")
    print(f"Utilization Rate: {fall_prediction['utilization_rate']}%")
    print(f"Sector: {fall_prediction['sector_info']['sector_name']}")
    print(f"Sector Description: {fall_prediction['sector_info']['sector_description']}")

def merge_weather_and_shelter_data(data_dir='data', output_file='merged_shelter_weather.csv'):
    """
    Load, process, and merge weather and shelter data by year, month, and day.
    Returns merged DataFrame and saves to CSV.
    """
    print("=" * 60)
    print("MERGING WEATHER AND SHELTER DATA")
    print("=" * 60)

    # --- Load weather data ---
    print("Loading weather data...")
    all_weather_data = []
    for year in [2017, 2018, 2019, 2020]:
        weather_dir = os.path.join(data_dir, f'Weather {year}')
        if os.path.exists(weather_dir):
            print(f"Processing {year}...")
            for month in range(1, 13):
                filename = f"en_climate_hourly_ON_6158359_{month:02d}-{year}_P1H.csv"
                filepath = os.path.join(weather_dir, filename)
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        df = df[['Date/Time', 'Temp (°C)', 'Year', 'Month', 'Day']]
                        all_weather_data.append(df)
                        print(f"  Loaded {filename}")
                    except Exception as e:
                        print(f"  Error loading {filename}: {e}")
    if not all_weather_data:
        print("No weather data found!")
        return None
    weather_df = pd.concat(all_weather_data, ignore_index=True)
    weather_df['Date/Time'] = pd.to_datetime(weather_df['Date/Time'])
    weather_df['year'] = weather_df['Date/Time'].dt.year
    weather_df['month'] = weather_df['Date/Time'].dt.month
    weather_df['day'] = weather_df['Date/Time'].dt.day
    weather_df['Temp (°C)'] = pd.to_numeric(weather_df['Temp (°C)'], errors='coerce')
    weather_df['OCCUPANCY_DATE'] = weather_df['Date/Time'].dt.strftime('%Y-%m-%dT00:00:00')
    daily_weather = weather_df.groupby(['OCCUPANCY_DATE', 'year', 'month', 'day']).agg({'Temp (°C)': 'mean'}).reset_index()
    daily_weather = daily_weather.rename(columns={'Temp (°C)': 'temp_mean'})
    print(f"Daily weather data shape: {daily_weather.shape}")
    print(f"Date range: {daily_weather['OCCUPANCY_DATE'].min()} to {daily_weather['OCCUPANCY_DATE'].max()}")

    # --- Load shelter data ---
    print("Loading shelter data...")
    all_shelter_data = []
    for year in [2017, 2018, 2019, 2020]:
        filename = f"Daily shelter occupancy {year}.csv"
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                all_shelter_data.append(df)
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    if not all_shelter_data:
        print("No shelter data found!")
        return None
    shelter_df = pd.concat(all_shelter_data, ignore_index=True)
    shelter_df['OCCUPANCY_DATE'] = pd.to_datetime(shelter_df['OCCUPANCY_DATE'], errors='coerce')
    shelter_df = shelter_df.dropna(subset=['OCCUPANCY_DATE'])
    shelter_df['OCCUPANCY_DATE'] = shelter_df['OCCUPANCY_DATE'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    print(f"Combined shelter data shape: {shelter_df.shape}")

    # --- Merge ---
    print("\nMerging data on OCCUPANCY_DATE...")
    merged_data = shelter_df.merge(daily_weather[['OCCUPANCY_DATE', 'temp_mean', 'month', 'day']], on='OCCUPANCY_DATE', how='left')
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Temperature data available for {merged_data['temp_mean'].notna().sum()} records")
    print(f"Missing temperature data for {merged_data['temp_mean'].isna().sum()} records")
    if merged_data['temp_mean'].isna().sum() > 0:
        print("Filling missing temperature data...")
        monthly_avg = merged_data.groupby('month')['temp_mean'].mean()
        merged_data['temp_mean'] = merged_data['temp_mean'].fillna(
            merged_data['month'].map(monthly_avg)
        )
        print(f"After filling: {merged_data['temp_mean'].isna().sum()} missing values")
    print("\nSample of merged data:")
    sample_cols = ['OCCUPANCY_DATE', 'SHELTER_NAME', 'OCCUPANCY', 'CAPACITY', 'temp_mean', 'month', 'day']
    available_cols = [col for col in sample_cols if col in merged_data.columns]
    print(merged_data[available_cols].head(10))
    merged_data.to_csv(output_file, index=False)
    print(f"\nMerged data saved to: {output_file}")
    return merged_data

# Call the merge function so merged data is available when running the notebook
merged_df = merge_weather_and_shelter_data() 