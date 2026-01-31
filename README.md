# MLOps Project - Loan Risk Prediction

This project implements a machine learning pipeline for loan risk prediction with comprehensive MLflow tracking and monitoring.

## Features

- **Dual Model Training**: Random Forest and Logistic Regression classifiers
- **MLflow Integration**: Complete experiment tracking with metrics, parameters, and artifacts
- **Comprehensive Testing**: Unit tests covering all components with pytest
- **Streamlit Web App**: Interactive web interface for predictions
- **Data Processing**: Automated preprocessing with label encoding for categorical features

## Project Structure

```
mlops/
├── data/                 # Training data
│   └── loan_risk_data.csv
├── src/                  # Source code
│   ├── app.py           # Streamlit web application
│   ├── model.py         # Model training with MLflow
│   ├── model1.joblib    # Random Forest model
│   ├── model2.joblib    # Logistic Regression model
│   └── encoders.joblib  # Label encoders
├── tests/                # Unit tests
│   ├── conftest.py      # Pytest fixtures
│   └── test_model.py    # Model tests
└── requirements.txt      # Dependencies

```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models

```bash
cd src
python model.py
```

This will:
- Train both Random Forest and Logistic Regression models
- Log all experiments to MLflow
- Save models and encoders
- Display performance metrics

### Viewing MLflow UI

```bash
cd src
mlflow ui
```

Then open `http://localhost:5000` in your browser to view:
- Model comparison metrics
- Parameter tuning history
- Confusion matrices
- Feature importance (Random Forest)

### Running the Web App

```bash
cd src
streamlit run app.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/test_model.py::TestLoadData -v
```

## Model Performance

The project trains two models and logs comprehensive metrics:

**Metrics Tracked:**
- Accuracy
- Precision (weighted and per-class)
- Recall (weighted and per-class)
- F1-Score (weighted and per-class)
- Confusion Matrix
- Classification Report

**Random Forest Additional Metrics:**
- Feature Importance

## MLflow Experiment Structure

The project creates separate runs for:
1. **Random_Forest_Classifier**: RF model training and metrics
2. **Logistic_Regression**: LR model training and metrics
3. **Data_Preprocessing**: Encoding and preprocessing parameters

## Testing

The test suite includes:
- Data loading tests
- Preprocessing validation
- Model training verification
- Encoder persistence tests
- End-to-end integration tests

Coverage target: >80%

## Risk Categories

The model predicts three risk categories:
- **Low Risk**: Low probability of default
- **Medium Risk**: Moderate probability of default
- **High Risk**: High probability of default

## Features Used

- Age
- Income
- Employment Type (Salaried/Self-employed/Unemployed)
- Residence Type (Owned/Rented/Parental Home)
- Credit Score
- Loan Amount
- Loan Term (months)
- Previous Default (Yes/No)

## License

MIT License