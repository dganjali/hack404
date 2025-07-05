#!/usr/bin/env python3
"""
Train and save the ML model for deployment using real_features.json
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

print("🚀 Starting verbose model training...")

# Load features from the real data
features_path = os.path.join(os.path.dirname(__file__), '../data/real_features.json')
print(f"📁 Loading features from: {features_path}")

with open(features_path, 'r') as f:
    features = json.load(f)

print(f"📊 Loaded {len(features)} feature records")

# Convert to DataFrame
features_df = pd.DataFrame(features)
print(f"📋 DataFrame shape: {features_df.shape}")
print(f"📋 DataFrame columns: {list(features_df.columns)}")

# Data exploration
print("\n🔍 Data Exploration:")
print(f"   • Date range: {features_df['date'].min()} to {features_df['date'].max()}")
print(f"   • Total occupancy range: {features_df['total_occupancy'].min()} to {features_df['total_occupancy'].max()}")
print(f"   • Average occupancy: {features_df['total_occupancy'].mean():.1f}")
print(f"   • Utilization rate range: {features_df['utilization_rate'].min():.3f} to {features_df['utilization_rate'].max():.3f}")
print(f"   • Average utilization: {features_df['utilization_rate'].mean():.3f}")

# Check for missing values
missing_values = features_df.isnull().sum()
if missing_values.sum() > 0:
    print(f"⚠️  Missing values found:")
    for col, count in missing_values.items():
        if count > 0:
            print(f"   • {col}: {count}")
else:
    print("✅ No missing values found")

# Drop rows with missing or invalid target
initial_rows = len(features_df)
features_df = features_df[features_df['total_occupancy'] > 0]
final_rows = len(features_df)
print(f"📊 Filtered data: {initial_rows} -> {final_rows} rows (removed {initial_rows - final_rows} invalid records)")

# Fill any remaining NaNs
features_df = features_df.fillna(0)
print("✅ Filled any remaining NaN values with 0")

# Define features and target
target_col = 'total_occupancy'
feature_cols = [
    'avg_occupancy', 'max_occupancy', 'min_occupancy', 'std_occupancy',
    'total_capacity', 'avg_capacity', 'max_capacity', 'min_capacity',
    'shelter_count', 'org_count', 'utilization_rate',
    'day_of_week', 'month', 'year',
    'is_weekend', 'is_winter', 'is_summer', 'is_spring', 'is_fall'
]

print(f"\n🎯 Target variable: {target_col}")
print(f"📈 Feature variables ({len(feature_cols)}): {feature_cols}")

X = features_df[feature_cols]
y = features_df[target_col]

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target vector shape: {y.shape}")

# Feature statistics
print("\n📈 Feature Statistics:")
for col in feature_cols:
    print(f"   • {col}: mean={X[col].mean():.2f}, std={X[col].std():.2f}, min={X[col].min():.2f}, max={X[col].max():.2f}")

# Split into train/test
print(f"\n🔀 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   • Training set: {X_train.shape[0]} samples")
print(f"   • Test set: {X_test.shape[0]} samples")

# Train model
print(f"\n🤖 Training RandomForest model...")
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    verbose=1    # Enable verbose output
)

print("   • Fitting model...")
model.fit(X_train, y_train)
print("   ✅ Model training complete!")

# Cross-validation
print(f"\n🔄 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"   • CV R² scores: {cv_scores}")
print(f"   • Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Evaluate on test set
print(f"\n🧪 Evaluating on test set...")
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"   • R² Score: {r2:.3f}")
print(f"   • Mean Absolute Error: {mae:.2f}")
print(f"   • Mean Squared Error: {mse:.2f}")
print(f"   • Root Mean Squared Error: {rmse:.2f}")

# Feature importance
print(f"\n🎯 Feature Importance:")
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_df.iterrows():
    print(f"   • {row['feature']}: {row['importance']:.3f}")

# Sample predictions
print(f"\n📊 Sample Predictions (first 10 test samples):")
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    print(f"   • Actual: {actual:.0f}, Predicted: {predicted:.0f}, Error: {error:.0f}")

# Save model
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
print(f"\n💾 Saving model to: {model_path}")
joblib.dump(model, model_path)
print("   ✅ Model saved successfully!")

# Model summary
print(f"\n🎉 Training Summary:")
print(f"   • Model: RandomForestRegressor")
print(f"   • Training samples: {X_train.shape[0]}")
print(f"   • Test samples: {X_test.shape[0]}")
print(f"   • Features: {len(feature_cols)}")
print(f"   • Best CV R²: {cv_scores.max():.3f}")
print(f"   • Test R²: {r2:.3f}")
print(f"   • Test MAE: {mae:.2f}")
print(f"   • Most important feature: {feature_importance_df.iloc[0]['feature']}")

print("\n✅ Model training complete and ready for deployment!") 