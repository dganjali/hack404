#!/usr/bin/env python3
"""
Simple script to merge weather temperature data with shelter occupancy data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_weather_data(data_dir='data'):
    """Load all weather data and extract daily temperature averages"""
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
                        # Only keep essential columns
                        df = df[['Date/Time', 'Temp (°C)', 'Year', 'Month', 'Day']]
                        all_weather_data.append(df)
                        print(f"  Loaded {filename}")
                    except Exception as e:
                        print(f"  Error loading {filename}: {e}")
    
    if not all_weather_data:
        print("No weather data found!")
        return None
    
    # Combine all weather data
    weather_df = pd.concat(all_weather_data, ignore_index=True)
    print(f"Combined weather data shape: {weather_df.shape}")
    
    # Convert date and extract date components
    weather_df['Date/Time'] = pd.to_datetime(weather_df['Date/Time'])
    weather_df['year'] = weather_df['Date/Time'].dt.year
    weather_df['month'] = weather_df['Date/Time'].dt.month
    weather_df['day'] = weather_df['Date/Time'].dt.day
    
    # Convert temperature to numeric
    weather_df['Temp (°C)'] = pd.to_numeric(weather_df['Temp (°C)'], errors='coerce')
    
    # Create OCCUPANCY_DATE in ISO format to match shelter data (YYYY-MM-DDT00:00:00)
    weather_df['OCCUPANCY_DATE'] = weather_df['Date/Time'].dt.strftime('%Y-%m-%dT00:00:00')
    
    # Calculate daily temperature averages
    daily_weather = weather_df.groupby(['OCCUPANCY_DATE', 'year', 'month', 'day']).agg({
        'Temp (°C)': 'mean'
    }).reset_index()
    
    # Rename temperature column
    daily_weather = daily_weather.rename(columns={'Temp (°C)': 'temp_mean'})
    
    print(f"Daily weather data shape: {daily_weather.shape}")
    print(f"Date range: {daily_weather['OCCUPANCY_DATE'].min()} to {daily_weather['OCCUPANCY_DATE'].max()}")
    
    return daily_weather

def load_shelter_data(data_dir='data'):
    """Load all shelter occupancy data"""
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
    
    # Combine all shelter data
    shelter_df = pd.concat(all_shelter_data, ignore_index=True)
    print(f"Combined shelter data shape: {shelter_df.shape}")
    
    # Convert date - handle mixed formats
    shelter_df['OCCUPANCY_DATE'] = pd.to_datetime(shelter_df['OCCUPANCY_DATE'], errors='coerce')
    shelter_df['date'] = shelter_df['OCCUPANCY_DATE'].dt.date
    
    # Drop rows where date could not be parsed
    shelter_df = shelter_df.dropna(subset=['date'])
    
    print(f"Date range: {shelter_df['date'].min()} to {shelter_df['date'].max()}")
    
    return shelter_df

def merge_weather_shelter_data():
    """Merge weather and shelter data by OCCUPANCY_DATE (ISO format)"""
    print("=" * 60)
    print("MERGING WEATHER AND SHELTER DATA")
    print("=" * 60)
    
    # Load data
    weather_data = load_weather_data()
    shelter_data = load_shelter_data()
    
    if weather_data is None or shelter_data is None:
        print("Failed to load data!")
        return None
    
    print("\nMerging data on OCCUPANCY_DATE...")
    
    # Convert shelter OCCUPANCY_DATE to string in ISO format (YYYY-MM-DDT00:00:00)
    shelter_data['OCCUPANCY_DATE'] = pd.to_datetime(shelter_data['OCCUPANCY_DATE'], errors='coerce')
    shelter_data['OCCUPANCY_DATE'] = shelter_data['OCCUPANCY_DATE'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Merge on OCCUPANCY_DATE
    merged_data = shelter_data.merge(weather_data[['OCCUPANCY_DATE', 'temp_mean', 'month', 'day']], on='OCCUPANCY_DATE', how='left')
    
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Temperature data available for {merged_data['temp_mean'].notna().sum()} records")
    print(f"Missing temperature data for {merged_data['temp_mean'].isna().sum()} records")
    
    # Fill missing temperature with monthly averages
    if merged_data['temp_mean'].isna().sum() > 0:
        print("Filling missing temperature data...")
        monthly_avg = merged_data.groupby('month')['temp_mean'].mean()
        merged_data['temp_mean'] = merged_data['temp_mean'].fillna(
            merged_data['month'].map(monthly_avg)
        )
        print(f"After filling: {merged_data['temp_mean'].isna().sum()} missing values")
    
    # Show sample of merged data
    print("\nSample of merged data:")
    sample_cols = ['OCCUPANCY_DATE', 'SHELTER_NAME', 'OCCUPANCY', 'CAPACITY', 'temp_mean', 'month', 'day']
    available_cols = [col for col in sample_cols if col in merged_data.columns]
    print(merged_data[available_cols].head(10))
    
    # Save merged data
    output_file = 'merged_shelter_weather.csv'
    merged_data.to_csv(output_file, index=False)
    print(f"\nMerged data saved to: {output_file}")
    
    return merged_data

if __name__ == "__main__":
    merged_data = merge_weather_shelter_data() 