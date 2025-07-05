#!/usr/bin/env python3
"""
Training script for the Shelter Occupancy Prediction Model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import ShelterDataPreprocessor
from shelter_model import ShelterOccupancyModel

def plot_training_history(history, save_path='models/training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(y_true, y_pred, save_path='models/predictions.png'):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Occupancy')
    plt.ylabel('Predicted Occupancy')
    plt.title('Actual vs Predicted Occupancy')
    plt.grid(True)
    
    # Residuals
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred.flatten()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Occupancy')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    # Distribution of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    
    # Time series of predictions (first 100 samples)
    plt.subplot(2, 2, 4)
    n_samples = min(100, len(y_true))
    plt.plot(y_true[:n_samples], label='Actual', alpha=0.7)
    plt.plot(y_pred[:n_samples], label='Predicted', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Occupancy')
    plt.title('Time Series Comparison (First 100 samples)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_performance(y_true, y_pred):
    """Evaluate model performance with various metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate percentage error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print("="*50)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def main():
    """Main training function"""
    print("Starting Shelter Occupancy Prediction Model Training")
    print("="*60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Data Preprocessing
    print("\n1. Loading and preprocessing data...")
    preprocessor = ShelterDataPreprocessor()
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(sequence_length=30)
        print(f"✓ Data preprocessing completed!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Number of features: {X_train.shape[-1]}")
        
        # Save preprocessors
        preprocessor.save_preprocessors()
        
    except Exception as e:
        print(f"✗ Error during data preprocessing: {e}")
        return
    
    # Step 2: Model Training
    print("\n2. Training the model...")
    
    # Initialize model
    n_features = X_train.shape[-1]
    model = ShelterOccupancyModel(
        sequence_length=30,
        n_features=n_features
    )
    
    # Train different model architectures
    model_types = ['lstm', 'conv_lstm']  # 'attention' can be added if needed
    
    best_model = None
    best_score = float('inf')
    best_model_type = None
    
    for model_type in model_types:
        print(f"\n   Training {model_type.upper()} model...")
        
        try:
            # Train the model
            history = model.train(
                X_train, y_train,
                X_test, y_test,
                epochs=10,  # Reduced to 10 epochs
                batch_size=32,
                model_type=model_type
            )
            
            # Evaluate the model
            predictions = model.predict(X_test)
            metrics = evaluate_model_performance(y_test, predictions)
            
            # Check if this is the best model
            if metrics['mae'] < best_score:
                best_score = metrics['mae']
                best_model = model
                best_model_type = model_type
            
            # Plot training history
            plot_training_history(history, f'models/training_history_{model_type}.png')
            
            # Plot predictions
            plot_predictions(y_test, predictions, f'models/predictions_{model_type}.png')
            
            print(f"   ✓ {model_type.upper()} model training completed!")
            
        except Exception as e:
            print(f"   ✗ Error training {model_type} model: {e}")
            continue
    
    # Step 3: Save the best model
    if best_model is not None:
        print(f"\n3. Saving best model ({best_model_type.upper()})...")
        best_model.save_model(f'models/shelter_model_{best_model_type}.h5')
        
        # Save model info
        model_info = {
            'model_type': best_model_type,
            'sequence_length': 30,
            'n_features': n_features,
            'best_mae': best_score,
            'training_date': datetime.now().isoformat(),
            'data_shape': {
                'X_train': X_train.shape,
                'X_test': X_test.shape,
                'y_train': y_train.shape,
                'y_test': y_test.shape
            }
        }
        
        import json
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✓ Model training completed successfully!")
        print(f"   Best model: {best_model_type.upper()}")
        print(f"   Best MAE: {best_score:.2f}")
        print(f"   Model saved to: models/shelter_model_{best_model_type}.h5")
        
    else:
        print("✗ No models were successfully trained.")
    
    print("\n" + "="*60)
    print("Training completed!")

if __name__ == "__main__":
    main() 