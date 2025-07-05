#!/usr/bin/env python3
"""
Training script for the Shelter Prediction Model
This script can be run independently to train the model with custom parameters.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_service.model import ShelterMLService

def main():
    parser = argparse.ArgumentParser(description='Train the Shelter Prediction Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length for LSTM')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    print("Starting Shelter Prediction Model Training")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"LSTM Layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Initialize ML service
        service = ShelterMLService(model_dir=args.output_dir)
        
        # Load data
        print("Loading data...")
        if not service.load_data():
            print("Failed to load data. Exiting.")
            return 1
        
        # Train model
        print("Training model...")
        success = service.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        if success:
            print("Training completed successfully!")
            print(f"Model saved to: {args.output_dir}")
            
            # Test prediction
            print("Testing prediction...")
            test_prediction = service.predict_shelter("shelter_1874", days=3)
            if test_prediction:
                print("Test prediction successful!")
                print(f"Shelter: {test_prediction['shelter_name']}")
                print(f"Predictions: {len(test_prediction['predictions'])} days")
            else:
                print("Test prediction failed!")
            
            return 0
        else:
            print("Training failed!")
            return 1
            
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 