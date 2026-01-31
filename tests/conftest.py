import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    data = {
        'Age': [30, 45, 25, 50, 35, 40, 28, 55, 32, 38],
        'Income': [50000, 80000, 35000, 90000, 60000, 75000, 45000, 95000, 55000, 70000],
        'EmploymentType': ['Salaried', 'Self-employed', 'Salaried', 'Salaried', 'Self-employed', 
                          'Salaried', 'Unemployed', 'Salaried', 'Salaried', 'Self-employed'],
        'ResidenceType': ['Owned', 'Rented', 'Parental Home', 'Owned', 'Rented',
                         'Owned', 'Rented', 'Owned', 'Parental Home', 'Rented'],
        'CreditScore': [650, 720, 580, 750, 680, 700, 520, 780, 640, 710],
        'LoanAmount': [20000, 35000, 15000, 40000, 25000, 30000, 18000, 45000, 22000, 32000],
        'LoanTerm': [36, 48, 24, 60, 36, 48, 24, 60, 36, 48],
        'PreviousDefault': ['No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No'],
        'RiskCategory': ['Medium Risk', 'Low Risk', 'High Risk', 'Low Risk', 'Medium Risk',
                        'Medium Risk', 'High Risk', 'Low Risk', 'Medium Risk', 'Medium Risk']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(tmp_path, sample_data):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def temp_output_dir(tmp_path, monkeypatch):
    """Change working directory to temp directory for model outputs."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
