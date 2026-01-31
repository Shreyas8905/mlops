"""Simple test script to verify model training works."""
import sys
sys.path.insert(0, 'src')

from model import load_data, preprocess_data
import pandas as pd

print("Testing data loading and preprocessing...")

# Test load
df = load_data('data/loan_risk_data.csv')
print(f"✓ Data loaded: {df.shape}")

# Test preprocess
X, y, encoders = preprocess_data(df)
print(f"✓ Data preprocessed: X shape={X.shape}, y shape={y.shape}")
print(f"✓ Encoders created: {list(encoders.keys())}")

print("\nAll basic tests passed!")
