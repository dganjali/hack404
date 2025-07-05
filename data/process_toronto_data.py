#!/usr/bin/env python3
"""
Toronto Shelter Data Processor
Extracts features from Toronto shelter system CSV and converts to our format
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any

class TorontoDataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.processed_data = {}
        
    def load_data(self):
        """Load and clean the CSV data"""
        print("Loading Toronto shelter data...")
        self.df = pd.read_csv(self.csv_path)
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Parse dates
        self.df['date'] = pd.to_datetime(self.df['date(mmm-yy)'], format='%b-%y')
        
        print(f"Loaded {len(self.df)} records")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
    def extract_features(self):
        """Extract meaningful features for ML prediction"""
        print("Extracting features...")
        
        # Group by date and population group
        features = []
        
        for date in self.df['date'].unique():
            date_data = self.df[self.df['date'] == date]
            
            # Aggregate across all population groups for this date
            daily_features = {
                'date': date.strftime('%Y-%m-%d'),
                'total_actively_homeless': int(date_data['actively_homeless'].sum()),
                'total_newly_identified': int(date_data['newly_identified'].sum()),
                'total_moved_to_housing': int(date_data['moved_to_housing'].sum()),
                'total_returned_to_shelter': int(date_data['returned_to_shelter'].sum()),
                'total_became_inactive': int(date_data['became_inactive'].sum()),
                
                # Demographic breakdown
                'total_under16': int(date_data['ageunder16'].sum()),
                'total_16_24': int(date_data['age16-24'].sum()),
                'total_25_34': int(date_data['age25-34'].sum()),
                'total_35_44': int(date_data['age35-44'].sum()),
                'total_45_54': int(date_data['age45-54'].sum()),
                'total_55_64': int(date_data['age55-64'].sum()),
                'total_65_over': int(date_data['age65over'].sum()),
                
                # Gender breakdown
                'total_male': int(date_data['gender_male'].sum()),
                'total_female': int(date_data['gender_female'].sum()),
                'total_transgender': int(date_data['gender_transgender,non-binary_or_two_spirit'].sum()),
            }
            
            # Extract population group percentages more safely
            try:
                chronic_data = date_data[date_data['population_group'] == 'Chronic']
                daily_features['chronic_percentage'] = float(chronic_data['population_group_percentage'].iloc[0].rstrip('%')) if len(chronic_data) > 0 else 0.0
            except:
                daily_features['chronic_percentage'] = 0.0
                
            try:
                refugee_data = date_data[date_data['population_group'] == 'Refugees']
                daily_features['refugee_percentage'] = float(refugee_data['population_group_percentage'].iloc[0].rstrip('%')) if len(refugee_data) > 0 else 0.0
            except:
                daily_features['refugee_percentage'] = 0.0
                
            try:
                family_data = date_data[date_data['population_group'] == 'Families']
                daily_features['family_percentage'] = float(family_data['population_group_percentage'].iloc[0].rstrip('%')) if len(family_data) > 0 else 0.0
            except:
                daily_features['family_percentage'] = 0.0
                
            try:
                youth_data = date_data[date_data['population_group'] == 'Youth']
                daily_features['youth_percentage'] = float(youth_data['population_group_percentage'].iloc[0].rstrip('%')) if len(youth_data) > 0 else 0.0
            except:
                daily_features['youth_percentage'] = 0.0
                
            try:
                single_adult_data = date_data[date_data['population_group'] == 'Single Adult']
                daily_features['single_adult_percentage'] = float(single_adult_data['population_group_percentage'].iloc[0].rstrip('%')) if len(single_adult_data) > 0 else 0.0
            except:
                daily_features['single_adult_percentage'] = 0.0
            
            # Calculate derived features
            daily_features['net_flow'] = daily_features['total_newly_identified'] - daily_features['total_moved_to_housing']
            daily_features['turnover_rate'] = (daily_features['total_moved_to_housing'] + daily_features['total_returned_to_shelter']) / max(daily_features['total_actively_homeless'], 1)
            daily_features['housing_success_rate'] = daily_features['total_moved_to_housing'] / max(daily_features['total_newly_identified'], 1)
            
            features.append(daily_features)
        
        # Sort by date
        features.sort(key=lambda x: x['date'])
        return features
    
    def create_shelter_data(self, features: List[Dict]) -> List[Dict]:
        """Create shelter data based on Toronto shelter system"""
        shelters = [
            {
                "id": "toronto_shelter_1",
                "name": "Toronto Central Shelter",
                "capacity": 200,
                "current_beds": int(features[-1]['total_actively_homeless'] * 0.25),
                "current_meals": int(features[-1]['total_actively_homeless'] * 0.25 * 2.5),
                "current_kits": int(features[-1]['total_actively_homeless'] * 0.25 * 0.8)
            },
            {
                "id": "toronto_shelter_2", 
                "name": "North York Refuge",
                "capacity": 150,
                "current_beds": int(features[-1]['total_actively_homeless'] * 0.2),
                "current_meals": int(features[-1]['total_actively_homeless'] * 0.2 * 2.5),
                "current_kits": int(features[-1]['total_actively_homeless'] * 0.2 * 0.8)
            },
            {
                "id": "toronto_shelter_3",
                "name": "Scarborough Safe Haven",
                "capacity": 120,
                "current_beds": int(features[-1]['total_actively_homeless'] * 0.15),
                "current_meals": int(features[-1]['total_actively_homeless'] * 0.15 * 2.5),
                "current_kits": int(features[-1]['total_actively_homeless'] * 0.15 * 0.8)
            },
            {
                "id": "toronto_shelter_4",
                "name": "Etobicoke Community Shelter",
                "capacity": 100,
                "current_beds": int(features[-1]['total_actively_homeless'] * 0.12),
                "current_meals": int(features[-1]['total_actively_homeless'] * 0.12 * 2.5),
                "current_kits": int(features[-1]['total_actively_homeless'] * 0.12 * 0.8)
            },
            {
                "id": "toronto_shelter_5",
                "name": "East York Family Shelter",
                "capacity": 80,
                "current_beds": int(features[-1]['total_actively_homeless'] * 0.1),
                "current_meals": int(features[-1]['total_actively_homeless'] * 0.1 * 2.5),
                "current_kits": int(features[-1]['total_actively_homeless'] * 0.1 * 0.8)
            }
        ]
        return shelters
    
    def create_intake_history(self, features: List[Dict]) -> List[Dict]:
        """Create intake history based on Toronto data patterns"""
        history = []
        
        # Use the last 30 days of data for each shelter
        recent_features = features[-30:] if len(features) >= 30 else features
        
        for shelter_id in [f"toronto_shelter_{i}" for i in range(1, 6)]:
            shelter_ratio = {
                "toronto_shelter_1": 0.25,
                "toronto_shelter_2": 0.20,
                "toronto_shelter_3": 0.15,
                "toronto_shelter_4": 0.12,
                "toronto_shelter_5": 0.10
            }[shelter_id]
            
            for feature in recent_features:
                # Base demand on actively homeless and demographic factors
                base_demand = feature['total_actively_homeless'] * shelter_ratio
                
                # Add seasonal and demographic variations
                date_obj = datetime.strptime(feature['date'], '%Y-%m-%d')
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * date_obj.timetuple().tm_yday / 365)
                
                # Youth factor (higher demand for youth shelters)
                youth_factor = 1.0
                if shelter_id == "toronto_shelter_2":  # North York has more youth
                    youth_factor = 1.3
                
                # Family factor
                family_factor = 1.0
                if shelter_id == "toronto_shelter_5":  # East York Family Shelter
                    family_factor = 1.4
                
                beds_needed = int(base_demand * seasonal_factor * youth_factor * family_factor)
                meals_needed = int(beds_needed * 2.5)  # 2.5 meals per bed
                kits_needed = int(beds_needed * 0.8)   # 0.8 kits per bed
                
                history.append({
                    "shelter_id": shelter_id,
                    "date": feature['date'],
                    "beds_needed": max(0, beds_needed),
                    "meals_needed": max(0, meals_needed),
                    "kits_needed": max(0, kits_needed)
                })
        
        return history
    
    def process(self):
        """Main processing pipeline"""
        self.load_data()
        features = self.extract_features()
        
        # Create shelter data
        shelters = self.create_shelter_data(features)
        
        # Create intake history
        history = self.create_intake_history(features)
        
        # Save processed data
        with open("../backend/toronto_shelters.json", "w") as f:
            json.dump(shelters, f, indent=2)
        
        with open("../backend/toronto_intake_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # Save features for ML model
        with open("../backend/toronto_features.json", "w") as f:
            json.dump(features, f, indent=2)
        
        print(f"✅ Processed Toronto data:")
        print(f"   • {len(shelters)} shelters created")
        print(f"   • {len(history)} intake records generated")
        print(f"   • {len(features)} feature records extracted")
        print(f"   • Date range: {features[0]['date']} to {features[-1]['date']}")
        
        return {
            'shelters': shelters,
            'history': history,
            'features': features
        }

def main():
    processor = TorontoDataProcessor("toronto-shelter-system-flow.csv")
    result = processor.process()
    
    print("\n📊 Toronto Data Summary:")
    print(f"   • Total actively homeless: {result['features'][-1]['total_actively_homeless']}")
    print(f"   • Housing success rate: {result['features'][-1]['housing_success_rate']:.2%}")
    print(f"   • Net flow: {result['features'][-1]['net_flow']}")
    print(f"   • Turnover rate: {result['features'][-1]['turnover_rate']:.2f}")

if __name__ == "__main__":
    main() 