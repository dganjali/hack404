#!/usr/bin/env python3
"""
Training script for the Shelter Occupancy Prediction Model
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import ShelterDataPreprocessor
from shelter_model import ShelterPredictionModel

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

def train_shelter_model(model_type='lstm', epochs=50, batch_size=32):
    """Train the shelter prediction model using the new aggregated pipeline"""
    
    print("=" * 60)
    print("SHELTER OCCUPANCY PREDICTION MODEL TRAINING")
    print("=" * 60)
    print(f"Model Type: {model_type.upper()}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n1. DATA PREPROCESSING")
    print("-" * 30)
    
    preprocessor = ShelterDataPreprocessor()
    
    # File paths
    file_paths = [
        'data/Daily shelter occupancy 2017.csv',
        'data/Daily shelter occupancy 2018.csv',
        'data/Daily shelter occupancy 2019.csv',
        'data/Daily shelter occupancy 2020.csv'
    ]
    
    # Process data using the new pipeline
    processed_data = preprocessor.process_data(file_paths)
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    daily_data = processed_data['daily_data']
    scaler = processed_data['scaler']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target range: {y_train.min():.1f} - {y_train.max():.1f}")
    
    # Step 2: Model Training
    print("\n2. MODEL TRAINING")
    print("-" * 30)
    
    # Create and train model
    model = ShelterPredictionModel(model_type=model_type)
    model.scaler = scaler
    
    # Split training data for validation
    val_split = int(0.8 * len(X_train))
    X_train_split = X_train[:val_split]
    X_val_split = X_train[val_split:]
    y_train_split = y_train[:val_split]
    y_val_split = y_train[val_split:]
    
    print(f"Training samples: {len(X_train_split)}")
    print(f"Validation samples: {len(X_val_split)}")
    
    # Train the model
    history = model.train_model(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Step 3: Model Evaluation
    print("\n3. MODEL EVALUATION")
    print("-" * 30)
    
    # Evaluate on test set
    evaluation_results = model.evaluate_model(X_test, y_test)
    
    # Step 4: Save Model and Results
    print("\n4. SAVING MODEL AND RESULTS")
    print("-" * 30)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/shelter_model_{model_type}.h5'
    model.save_model(model_path)
    
    # Save scaler
    scaler_path = f'models/scaler_{model_type}.pkl'
    joblib.dump(scaler, scaler_path)
    
    # Save daily data for reference
    daily_data_path = f'models/daily_data_{model_type}.pkl'
    joblib.dump(daily_data, daily_data_path)
    
    # Save evaluation results
    results = {
        'model_type': model_type,
        'training_date': datetime.now().isoformat(),
        'evaluation_results': evaluation_results,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'daily_data_path': daily_data_path,
        'data_shape': {
            'X_train': X_train.shape,
            'X_test': X_test.shape,
            'y_train': y_train.shape,
            'y_test': y_test.shape
        }
    }
    
    results_path = f'models/training_results_{model_type}.pkl'
    joblib.dump(results, results_path)
    
    # Step 5: Generate Plots
    print("\n5. GENERATING PLOTS")
    print("-" * 30)
    
    # Create plots directory
    os.makedirs('models/plots', exist_ok=True)
    
    # Training history plot
    history_path = f'models/plots/training_history_{model_type}.png'
    model.plot_training_history(history, history_path)
    
    # Predictions plot
    predictions_path = f'models/plots/predictions_{model_type}.png'
    model.plot_predictions(
        evaluation_results['actual'],
        evaluation_results['predictions'],
        predictions_path
    )
    
    # Step 6: Summary
    print("\n6. TRAINING SUMMARY")
    print("-" * 30)
    print(f"Model Type: {model_type.upper()}")
    print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MAE: {evaluation_results['mae']:.2f}")
    print(f"RMSE: {evaluation_results['rmse']:.2f}")
    print(f"R²: {evaluation_results['r2']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Results saved to: {results_path}")
    
    # Step 7: Test Prediction
    print("\n7. TEST PREDICTION")
    print("-" * 30)
    
    # Test prediction for a sample shelter
    test_shelter = {
        'name': 'Test Shelter',
        'maxCapacity': 100
    }
    
    test_date = '2024-01-15'
    
    try:
        prediction = model.predict_for_shelter(test_shelter, test_date)
        print(f"Test Prediction for {test_date}:")
        print(f"  Shelter: {prediction['shelter_name']}")
        print(f"  Predicted Occupancy: {prediction['predicted_occupancy']}")
        print(f"  Max Capacity: {prediction['max_capacity']}")
        print(f"  Utilization Rate: {prediction['utilization_rate']}%")
    except Exception as e:
        print(f"Test prediction failed: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return model, results

def compare_models():
    """Compare different model types"""
    print("COMPARING DIFFERENT MODEL TYPES")
    print("=" * 60)
    
    model_types = ['lstm', 'conv_lstm', 'attention']
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        try:
            model, result = train_shelter_model(model_type=model_type, epochs=30)
            results[model_type] = result['evaluation_results']
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
            results[model_type] = None
    
    # Print comparison
    print("\nMODEL COMPARISON")
    print("-" * 60)
    print(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 60)
    
    for model_type, result in results.items():
        if result is not None:
            print(f"{model_type.upper():<15} {result['mae']:<10.2f} {result['rmse']:<10.2f} {result['r2']:<10.4f}")
        else:
            print(f"{model_type.upper():<15} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train shelter occupancy prediction model')
    parser.add_argument('--model-type', type=str, default='lstm', 
                       choices=['lstm', 'conv_lstm', 'attention'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all model types')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        train_shelter_model(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        ) 