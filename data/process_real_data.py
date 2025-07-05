#!/usr/bin/env python3
"""
Process real shelter occupancy data from multiple CSV files
Extract features and compile into a single dataset for ML training
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any

class RealDataProcessor:
    def __init__(self):
        self.data_dir = "."
        self.csv_files = [
            "Daily shelter occupancy 2017.csv",
            "Daily shelter occupancy 2018.csv", 
            "Daily shelter occupancy 2019.csv",
            "Daily shelter occupancy 2020.csv"
        ]
        self.combined_data = []
        self.features_data = []
        
    def load_all_csv_files(self):
        """Load and combine all CSV files"""
        print("📊 Loading real shelter occupancy data...")
        
        all_data = []
        
        for csv_file in self.csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            if os.path.exists(file_path):
                print(f"📁 Loading {csv_file}...")
                try:
                    df = pd.read_csv(file_path)
                    print(f"   • Shape: {df.shape}")
                    print(f"   • Columns: {list(df.columns)}")
                    all_data.append(df)
                except Exception as e:
                    print(f"   ❌ Error loading {csv_file}: {e}")
            else:
                print(f"   ⚠️  File not found: {csv_file}")
        
        if not all_data:
            raise ValueError("No CSV files could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Combined dataset shape: {combined_df.shape}")
        
        return combined_df
    
    def clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the combined dataset"""
        print("🧹 Cleaning and processing data...")
        
        # Display info about the data
        print(f"📋 Data info:")
        print(f"   • Columns: {list(df.columns)}")
        print(f"   • Data types: {df.dtypes.to_dict()}")
        print(f"   • Missing values: {df.isnull().sum().to_dict()}")
        
        # Convert OCCUPANCY_DATE to datetime
        try:
            df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'])
            print("   ✅ Converted OCCUPANCY_DATE to datetime")
        except Exception as e:
            print(f"   ⚠️  Could not convert OCCUPANCY_DATE to datetime: {e}")
        
        # Handle missing values
        if df['CAPACITY'].isnull().sum() > 0:
            median_capacity = df['CAPACITY'].median()
            df['CAPACITY'] = df['CAPACITY'].fillna(median_capacity)
            print(f"   ✅ Filled missing CAPACITY values with median: {median_capacity}")
        
        # Remove rows with invalid occupancy or capacity
        initial_rows = len(df)
        df = df[(df['OCCUPANCY'] >= 0) & (df['CAPACITY'] > 0)]
        final_rows = len(df)
        if initial_rows != final_rows:
            print(f"   ✅ Removed {initial_rows - final_rows} rows with invalid occupancy/capacity")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        final_rows = len(df)
        if initial_rows != final_rows:
            print(f"   ✅ Removed {initial_rows - final_rows} duplicate rows")
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> List[Dict]:
        """Extract features from the processed data"""
        print("🔍 Extracting features from real data...")
        
        features = []
        
        # Group by date to get daily aggregates
        daily_data = df.groupby('OCCUPANCY_DATE').agg({
            'OCCUPANCY': ['sum', 'mean', 'max', 'min', 'std'],
            'CAPACITY': ['sum', 'mean', 'max', 'min'],
            'SHELTER_NAME': 'nunique',  # Number of unique shelters
            'ORGANIZATION_NAME': 'nunique'  # Number of unique organizations
        }).reset_index()
        
        # Flatten column names
        daily_data.columns = ['date', 'total_occupancy', 'avg_occupancy', 'max_occupancy', 'min_occupancy', 'std_occupancy',
                            'total_capacity', 'avg_capacity', 'max_capacity', 'min_capacity', 'shelter_count', 'org_count']
        
        # Calculate utilization rate
        daily_data['utilization_rate'] = daily_data['total_occupancy'] / daily_data['total_capacity']
        
        # Extract features for each day
        for idx, row in daily_data.iterrows():
            try:
                # Robust date handling
                date_val = row['date']
                if isinstance(date_val, str):
                    try:
                        date_obj = pd.to_datetime(date_val)
                    except Exception:
                        date_obj = None
                elif hasattr(date_val, 'strftime'):
                    date_obj = date_val
                else:
                    date_obj = None
                if date_obj is not None:
                    date_str = date_obj.strftime('%Y-%m-%d')
                    day_of_week = date_obj.weekday()
                    month = date_obj.month
                    year = date_obj.year
                else:
                    date_str = str(date_val)
                    day_of_week = -1
                    month = -1
                    year = -1
                
                feature_record = {
                    'date': date_str,
                    'total_occupancy': int(row['total_occupancy']) if not pd.isna(row['total_occupancy']) else 0,
                    'avg_occupancy': float(row['avg_occupancy']) if not pd.isna(row['avg_occupancy']) else 0.0,
                    'max_occupancy': int(row['max_occupancy']) if not pd.isna(row['max_occupancy']) else 0,
                    'min_occupancy': int(row['min_occupancy']) if not pd.isna(row['min_occupancy']) else 0,
                    'std_occupancy': float(row['std_occupancy']) if not pd.isna(row['std_occupancy']) else 0.0,
                    'total_capacity': int(row['total_capacity']) if not pd.isna(row['total_capacity']) else 0,
                    'avg_capacity': float(row['avg_capacity']) if not pd.isna(row['avg_capacity']) else 0.0,
                    'max_capacity': int(row['max_capacity']) if not pd.isna(row['max_capacity']) else 0,
                    'min_capacity': int(row['min_capacity']) if not pd.isna(row['min_capacity']) else 0,
                    'shelter_count': int(row['shelter_count']) if not pd.isna(row['shelter_count']) else 0,
                    'org_count': int(row['org_count']) if not pd.isna(row['org_count']) else 0,
                    'utilization_rate': float(row['utilization_rate']) if not pd.isna(row['utilization_rate']) else 0.0,
                    'day_of_week': day_of_week,
                    'month': month,
                    'year': year,
                    'is_weekend': 1 if day_of_week >= 5 else 0,
                    'is_winter': 1 if month in [12, 1, 2] else 0,
                    'is_summer': 1 if month in [6, 7, 8] else 0,
                    'is_spring': 1 if month in [3, 4, 5] else 0,
                    'is_fall': 1 if month in [9, 10, 11] else 0,
                }
                features.append(feature_record)
            except Exception as e:
                print(f"⚠️  Error processing row {idx}: {e}")
                continue
        
        print(f"✅ Extracted {len(features)} feature records")
        return features
    
    def create_shelter_records(self, df: pd.DataFrame) -> List[Dict]:
        """Create shelter records from the data"""
        print("🏠 Creating shelter records...")
        
        shelters = []
        
        # Group by shelter name
        shelter_groups = df.groupby('SHELTER_NAME')
        
        for shelter_name, group in shelter_groups:
            try:
                # Calculate shelter statistics
                avg_occupancy = group['OCCUPANCY'].mean()
                max_occupancy = group['OCCUPANCY'].max()
                avg_capacity = group['CAPACITY'].mean()
                total_records = len(group)
                
                # Get organization info
                org_name = group['ORGANIZATION_NAME'].iloc[0]
                facility_name = group['FACILITY_NAME'].iloc[0]
                program_name = group['PROGRAM_NAME'].iloc[0]
                sector = group['SECTOR'].iloc[0]
                
                shelter_record = {
                    "id": f"shelter_{hash(shelter_name) % 10000}",
                    "name": shelter_name,
                    "organization": org_name,
                    "facility": facility_name,
                    "program": program_name,
                    "sector": sector,
                    "capacity": int(avg_capacity),
                    "current_beds": int(avg_occupancy),
                    "current_meals": int(avg_occupancy * 2.5),  # Estimate meals
                    "current_kits": int(avg_occupancy * 0.8),   # Estimate kits
                    "avg_occupancy": float(avg_occupancy),
                    "max_occupancy": int(max_occupancy),
                    "utilization_rate": float(avg_occupancy / avg_capacity) if avg_capacity > 0 else 0,
                    "total_records": total_records
                }
                shelters.append(shelter_record)
                
            except Exception as e:
                print(f"⚠️  Error processing shelter {shelter_name}: {e}")
                continue
        
        print(f"✅ Created {len(shelters)} shelter records")
        return shelters
    
    def create_intake_history(self, df: pd.DataFrame) -> List[Dict]:
        """Create intake history from the data"""
        print("📈 Creating intake history...")
        
        history = []
        
        # Group by date and shelter
        grouped = df.groupby(['OCCUPANCY_DATE', 'SHELTER_NAME'])
        
        for (date, shelter_name), group in grouped:
            try:
                # Handle date conversion properly
                if isinstance(date, str):
                    date_str = date
                elif hasattr(date, 'strftime'):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                
                # Calculate demand based on occupancy
                avg_occupancy = group['OCCUPANCY'].mean()
                capacity = group['CAPACITY'].mean()
                
                # Estimate demand (beds needed = current occupancy + some buffer)
                beds_needed = int(avg_occupancy * 1.1)  # 10% buffer
                meals_needed = int(beds_needed * 2.5)   # 2.5 meals per person
                kits_needed = int(beds_needed * 0.8)    # 0.8 kits per person
                
                record = {
                    "shelter_id": f"shelter_{hash(shelter_name) % 10000}",
                    "shelter_name": shelter_name,
                    "date": date_str,
                    "beds_needed": beds_needed,
                    "meals_needed": meals_needed,
                    "kits_needed": kits_needed,
                    "actual_occupancy": int(avg_occupancy),
                    "capacity": int(capacity),
                    "utilization_rate": float(avg_occupancy / capacity) if capacity > 0 else 0,
                    "organization": group['ORGANIZATION_NAME'].iloc[0],
                    "facility": group['FACILITY_NAME'].iloc[0],
                    "program": group['PROGRAM_NAME'].iloc[0],
                    "sector": group['SECTOR'].iloc[0]
                }
                history.append(record)
                
            except Exception as e:
                print(f"⚠️  Error processing group {date}, {shelter_name}: {e}")
                continue
        
        print(f"✅ Created {len(history)} intake history records")
        return history
    
    def save_processed_data(self):
        """Save all processed data to JSON files"""
        print("💾 Saving processed data...")
        
        # Save shelters
        with open("real_shelters.json", "w") as f:
            json.dump(self.combined_data['shelters'], f, indent=2)
        print("   ✅ Saved real_shelters.json")
        
        # Save intake history
        with open("real_intake_history.json", "w") as f:
            json.dump(self.combined_data['history'], f, indent=2)
        print("   ✅ Saved real_intake_history.json")
        
        # Save features
        with open("real_features.json", "w") as f:
            json.dump(self.combined_data['features'], f, indent=2)
        print("   ✅ Saved real_features.json")
        
        # Create summary
        summary = {
            "data_source": "Real Shelter Occupancy Data (2017-2020)",
            "shelters_count": len(self.combined_data['shelters']),
            "history_records": len(self.combined_data['history']),
            "features_records": len(self.combined_data['features']),
            "date_range": {
                "start": min([h['date'] for h in self.combined_data['history']]),
                "end": max([h['date'] for h in self.combined_data['history']])
            },
            "total_occupancy": sum([h['actual_occupancy'] for h in self.combined_data['history']]),
            "avg_utilization": np.mean([h['utilization_rate'] for h in self.combined_data['history']])
        }
        
        with open("real_data_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("   ✅ Saved real_data_summary.json")
        
        print(f"\n📊 Real Data Summary:")
        print(f"   • {summary['shelters_count']} shelters")
        print(f"   • {summary['history_records']} history records")
        print(f"   • {summary['features_records']} feature records")
        print(f"   • Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"   • Total occupancy tracked: {summary['total_occupancy']:,}")
        print(f"   • Average utilization: {summary['avg_utilization']:.1%}")
    
    def process(self):
        """Main processing pipeline"""
        print("🚀 Starting real data processing...")
        
        # Load all CSV files
        df = self.load_all_csv_files()
        
        # Clean and process data
        df = self.clean_and_process_data(df)
        
        # Extract features
        features = self.extract_features(df)
        
        # Create shelter records
        shelters = self.create_shelter_records(df)
        
        # Create intake history
        history = self.create_intake_history(df)
        
        # Store combined data
        self.combined_data = {
            'shelters': shelters,
            'history': history,
            'features': features
        }
        
        # Save processed data
        self.save_processed_data()
        
        print("✅ Real data processing complete!")

def main():
    processor = RealDataProcessor()
    processor.process()

if __name__ == "__main__":
    main() 