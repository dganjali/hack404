import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TorontoSectorMapper:
    """
    Toronto Sector Mapper for Shelter Location Features
    
    Implements the sector method for location-based prediction:
    1. Divides Toronto into geographic sectors
    2. Assigns shelters to sectors based on postal codes/addresses
    3. Creates sector-based features for prediction
    """
    
    def __init__(self):
        # Define Toronto sectors based on postal code areas (FSA - Forward Sortation Areas)
        # Toronto postal codes start with M1-M9
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
        
        self.sector_encoder = LabelEncoder()
        self.sector_encoder.fit(list(self.toronto_sectors.keys()))
        
    def get_sector_from_postal_code(self, postal_code: str) -> str:
        """Map postal code to sector"""
        if pd.isna(postal_code) or postal_code == '':
            return 'unknown'
        
        # Extract first 3 characters (FSA)
        fsa = str(postal_code).strip()[:3].upper()
        
        for sector_id, sector_info in self.toronto_sectors.items():
            if fsa in sector_info['postal_codes']:
                return sector_id
        
        return 'unknown'
    
    def get_sector_from_address(self, address: str, city: str, province: str) -> str:
        """Map address to sector using postal code extraction or address parsing"""
        # This is a simplified approach - in practice, you'd use geocoding APIs
        # For now, we'll rely on postal codes from the data
        
        # If we have a postal code in the address, extract it
        import re
        postal_pattern = r'[A-Z]\d[A-Z]\s?\d[A-Z]\d'
        postal_match = re.search(postal_pattern, address.upper())
        
        if postal_match:
            postal_code = postal_match.group()
            return self.get_sector_from_postal_code(postal_code)
        
        # If no postal code found, try to infer from address keywords
        address_lower = address.lower()
        
        # Downtown keywords
        if any(keyword in address_lower for keyword in ['downtown', 'financial', 'bay street', 'king street', 'queen street', 'university']):
            return 'downtown_core'
        
        # East end keywords
        if any(keyword in address_lower for keyword in ['scarborough', 'east york', 'beaches', 'danforth']):
            return 'east_end'
        
        # West end keywords
        if any(keyword in address_lower for keyword in ['parkdale', 'high park', 'junction', 'west toronto']):
            return 'west_end'
        
        # North end keywords
        if any(keyword in address_lower for keyword in ['north york', 'york', 'don mills', 'lawrence']):
            return 'north_end'
        
        # Etobicoke keywords
        if any(keyword in address_lower for keyword in ['etobicoke', 'rexdale', 'humber']):
            return 'etobicoke'
        
        return 'unknown'
    
    def assign_shelter_to_sector(self, shelter_data: pd.DataFrame) -> pd.DataFrame:
        """Assign each shelter to a sector"""
        print("Assigning shelters to sectors...")
        
        # Create a copy to avoid modifying original
        df = shelter_data.copy()
        
        # Add sector column
        df['sector'] = df.apply(
            lambda row: self.get_sector_from_postal_code(row['SHELTER_POSTAL_CODE']) 
            if pd.notna(row['SHELTER_POSTAL_CODE']) 
            else self.get_sector_from_address(row['SHELTER_ADDRESS'], row['SHELTER_CITY'], row['SHELTER_PROVINCE']),
            axis=1
        )
        
        # Encode sectors
        df['sector_encoded'] = self.sector_encoder.transform(df['sector'])
        
        # Add sector socioeconomic indicators
        for indicator in ['avg_income', 'population_density', 'transit_accessibility', 'crime_rate', 'homelessness_rate']:
            df[f'sector_{indicator}'] = df['sector'].map(
                {sector: self.sector_indicators[sector][indicator] 
                 for sector in self.sector_indicators.keys()}
            )
        
        return df
    
    def create_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # Create sector trend features (sector occupancy relative to historical average)
        df['sector_occupancy_ratio'] = df['OCCUPANCY'] / df['sector_avg_occupancy']
        df['sector_occupancy_zscore'] = (df['OCCUPANCY'] - df['sector_avg_occupancy']) / df['sector_std_occupancy']
        
        return df
    
    def get_sector_for_new_shelter(self, address: str, postal_code: str = None, city: str = 'Toronto', province: str = 'ON') -> Dict:
        """Get sector information for a new shelter"""
        if postal_code:
            sector_id = self.get_sector_from_postal_code(postal_code)
        else:
            sector_id = self.get_sector_from_address(address, city, province)
        
        sector_info = {
            'sector_id': sector_id,
            'sector_name': self.toronto_sectors.get(sector_id, {}).get('name', 'Unknown'),
            'sector_description': self.toronto_sectors.get(sector_id, {}).get('description', 'Unknown'),
            'socioeconomic_indicators': self.sector_indicators.get(sector_id, {}),
            'sector_encoded': self.sector_encoder.transform([sector_id])[0] if sector_id != 'unknown' else -1
        }
        
        return sector_info
    
    def create_sector_embedding_features(self, df: pd.DataFrame, embedding_dim: int = 8) -> pd.DataFrame:
        """Create sector embeddings for neural network models"""
        print("Creating sector embeddings...")
        
        # Simple embedding approach - in practice, you'd train proper embeddings
        sector_embeddings = {}
        
        for sector_id in self.toronto_sectors.keys():
            # Create embedding based on socioeconomic indicators
            indicators = self.sector_indicators.get(sector_id, {})
            
            # Normalize indicators to create embedding
            embedding = [
                indicators.get('avg_income', 0) / 100000,  # Normalize income
                indicators.get('population_density', 0) / 10000,  # Normalize density
                indicators.get('transit_accessibility', 0),
                indicators.get('crime_rate', 0),
                indicators.get('homelessness_rate', 0),
                0, 0, 0  # Padding to reach embedding_dim
            ]
            
            sector_embeddings[sector_id] = embedding[:embedding_dim]
        
        # Add embedding features to dataframe
        for i in range(embedding_dim):
            df[f'sector_embedding_{i}'] = df['sector'].map(
                {sector: embedding[i] for sector, embedding in sector_embeddings.items()}
            )
        
        return df
    
    def save_sector_mapper(self, filepath: str):
        """Save the sector mapper for later use"""
        mapper_data = {
            'toronto_sectors': self.toronto_sectors,
            'sector_indicators': self.sector_indicators,
            'sector_encoder': self.sector_encoder
        }
        joblib.dump(mapper_data, filepath)
        print(f"Sector mapper saved to {filepath}")
    
    def load_sector_mapper(self, filepath: str):
        """Load a saved sector mapper"""
        mapper_data = joblib.load(filepath)
        self.toronto_sectors = mapper_data['toronto_sectors']
        self.sector_indicators = mapper_data['sector_indicators']
        self.sector_encoder = mapper_data['sector_encoder']
        print(f"Sector mapper loaded from {filepath}")


class LocationFeaturePipeline:
    """
    Complete location feature pipeline for shelter prediction
    """
    
    def __init__(self):
        self.sector_mapper = TorontoSectorMapper()
        self.feature_columns = []
        
    def process_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process location features for the entire dataset"""
        print("Processing location features...")
        
        # Step 1: Assign shelters to sectors
        df = self.sector_mapper.assign_shelter_to_sector(df)
        
        # Step 2: Create sector-based features
        df = self.sector_mapper.create_sector_features(df)
        
        # Step 3: Create sector embeddings
        df = self.sector_mapper.create_sector_embedding_features(df)
        
        # Store feature column names
        self.feature_columns = [col for col in df.columns if any(prefix in col for prefix in 
                                                               ['sector_', 'shelters_in_sector', 'sector_occupancy'])]
        
        print(f"Created {len(self.feature_columns)} location features")
        return df
    
    def get_location_features_for_prediction(self, shelter_info: Dict) -> Dict:
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
        
        # Add sector embeddings
        for i in range(8):  # 8-dimensional embedding
            features[f'sector_embedding_{i}'] = 0  # Will be filled by the model
        
        return features
    
    def save_pipeline(self, filepath: str):
        """Save the location feature pipeline"""
        pipeline_data = {
            'sector_mapper': self.sector_mapper,
            'feature_columns': self.feature_columns
        }
        joblib.dump(pipeline_data, filepath)
        print(f"Location feature pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load a saved location feature pipeline"""
        pipeline_data = joblib.load(filepath)
        self.sector_mapper = pipeline_data['sector_mapper']
        self.feature_columns = pipeline_data['feature_columns']
        print(f"Location feature pipeline loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Test the sector mapper
    mapper = TorontoSectorMapper()
    
    # Test postal code mapping
    test_postal_codes = ['M5S 2P1', 'M6H 3Z5', 'M6G 3B1', 'M1B 1A1', 'M8V 1A1']
    for postal_code in test_postal_codes:
        sector = mapper.get_sector_from_postal_code(postal_code)
        print(f"Postal code {postal_code} -> Sector: {sector}")
    
    # Test address mapping
    test_addresses = [
        '100 Lippincott Street, Toronto, ON',
        '973 Lansdowne Avenue, Toronto, ON',
        '43 Christie Street, Toronto, ON'
    ]
    
    for address in test_addresses:
        sector = mapper.get_sector_from_address(address, 'Toronto', 'ON')
        print(f"Address {address} -> Sector: {sector}")
    
    print("\nSector mapping test completed!") 