import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import (
    load_data,
    preprocess_data,
    train_random_forest,
    train_logistic_regression
)


class TestLoadData:
    """Test cases for load_data function."""
    
    def test_load_data_returns_dataframe(self, sample_csv_path):
        """Test that load_data returns a pandas DataFrame."""
        df = load_data(sample_csv_path)
        assert isinstance(df, pd.DataFrame)
    
    def test_load_data_correct_shape(self, sample_csv_path):
        """Test that loaded data has correct shape."""
        df = load_data(sample_csv_path)
        assert df.shape[0] == 10  # 10 rows in sample data
        assert df.shape[1] == 9   # 9 columns
    
    def test_load_data_has_required_columns(self, sample_csv_path):
        """Test that all required columns are present."""
        df = load_data(sample_csv_path)
        required_columns = [
            'Age', 'Income', 'EmploymentType', 'ResidenceType',
            'CreditScore', 'LoanAmount', 'LoanTerm', 'PreviousDefault',
            'RiskCategory'
        ]
        for col in required_columns:
            assert col in df.columns
    
    def test_load_data_nonexistent_file(self):
        """Test that load_data raises error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")


class TestPreprocessData:
    """Test cases for preprocess_data function."""
    
    def test_preprocess_returns_three_items(self, sample_data):
        """Test that preprocess_data returns X, y, and encoders."""
        X, y, encoders = preprocess_data(sample_data)
        assert X is not None
        assert y is not None
        assert encoders is not None
    
    def test_preprocess_X_shape(self, sample_data):
        """Test that X has correct shape (all columns except target)."""
        X, y, encoders = preprocess_data(sample_data)
        assert X.shape[0] == 10  # 10 rows
        assert X.shape[1] == 8   # 8 features (9 columns - 1 target)
    
    def test_preprocess_y_shape(self, sample_data):
        """Test that y has correct shape."""
        X, y, encoders = preprocess_data(sample_data)
        assert len(y) == 10
        assert y.name == 'RiskCategory'
    
    def test_preprocess_categorical_encoded(self, sample_data):
        """Test that categorical columns are encoded as numeric."""
        X, y, encoders = preprocess_data(sample_data)
        categorical_cols = ['EmploymentType', 'ResidenceType', 'PreviousDefault']
        
        for col in categorical_cols:
            assert col in X.columns
            # Check if values are numeric
            assert pd.api.types.is_numeric_dtype(X[col])
    
    def test_preprocess_encoders_structure(self, sample_data):
        """Test that encoders dictionary has correct structure."""
        X, y, encoders = preprocess_data(sample_data)
        expected_keys = ['EmploymentType', 'ResidenceType', 'PreviousDefault']
        
        assert list(encoders.keys()) == expected_keys
        
        # Check that each encoder is a LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        for encoder in encoders.values():
            assert isinstance(encoder, LabelEncoder)
    
    def test_preprocess_encoders_can_transform(self, sample_data):
        """Test that encoders can transform original values."""
        X, y, encoders = preprocess_data(sample_data)
        
        # Test EmploymentType encoder
        assert 'Salaried' in encoders['EmploymentType'].classes_
        encoded_value = encoders['EmploymentType'].transform(['Salaried'])
        assert isinstance(encoded_value[0], (int, np.integer))
    
    def test_preprocess_no_missing_values_in_X(self, sample_data):
        """Test that X has no missing values after preprocessing."""
        X, y, encoders = preprocess_data(sample_data)
        assert X.isnull().sum().sum() == 0
    
    def test_preprocess_target_not_in_X(self, sample_data):
        """Test that RiskCategory is not in features."""
        X, y, encoders = preprocess_data(sample_data)
        assert 'RiskCategory' not in X.columns


class TestTrainRandomForest:
    """Test cases for train_random_forest function."""
    
    def test_train_rf_returns_model(self, sample_data, temp_output_dir):
        """Test that train_random_forest returns a model."""
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_random_forest(X_train, y_train, X_test, y_test, classes)
        
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
    
    def test_train_rf_can_predict(self, sample_data, temp_output_dir):
        """Test that trained model can make predictions."""
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_random_forest(X_train, y_train, X_test, y_test, classes)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_train_rf_saves_model_file(self, sample_data, temp_output_dir):
        """Test that model is saved to file."""
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_random_forest(X_train, y_train, X_test, y_test, classes)
        
        assert (temp_output_dir / 'model1.joblib').exists()
    
    def test_train_rf_model_parameters(self, sample_data, temp_output_dir):
        """Test that model has correct parameters."""
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_random_forest(X_train, y_train, X_test, y_test, classes)
        
        assert model.n_estimators == 100
        assert model.random_state == 42


class TestTrainLogisticRegression:
    """Test cases for train_logistic_regression function."""
    
    def test_train_lr_returns_model(self, sample_data, temp_output_dir):
        """Test that train_logistic_regression returns a model."""
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_logistic_regression(X_train, y_train, X_test, y_test, classes)
        
        from sklearn.linear_model import LogisticRegression
        assert isinstance(model, LogisticRegression)
    
    def test_train_lr_can_predict(self, sample_data, temp_output_dir):
        """Test that trained model can make predictions."""
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_logistic_regression(X_train, y_train, X_test, y_test, classes)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_train_lr_saves_model_file(self, sample_data, temp_output_dir):
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_logistic_regression(X_train, y_train, X_test, y_test, classes)
        
        assert (temp_output_dir / 'model2.joblib').exists()
    
    def test_train_lr_model_parameters(self, sample_data, temp_output_dir):
        """Test that model has correct parameters."""
        X, y, encoders = preprocess_data(sample_data)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        classes = sorted(y.unique())
        model = train_logistic_regression(X_train, y_train, X_test, y_test, classes)
        
        assert model.solver == 'lbfgs'
        assert model.max_iter == 1000
        assert model.random_state == 8888


class TestEncoderPersistence:
    """Test cases for encoder saving and loading."""
    
    def test_encoders_can_be_saved_and_loaded(self, sample_data, temp_output_dir):
        """Test that encoders can be saved and loaded correctly."""
        X, y, encoders = preprocess_data(sample_data)
        
        # Save encoders
        joblib.dump(encoders, temp_output_dir / 'encoders.joblib')
        
        # Load encoders
        loaded_encoders = joblib.load(temp_output_dir / 'encoders.joblib')
        
        assert loaded_encoders is not None
        assert list(loaded_encoders.keys()) == list(encoders.keys())
    
    def test_loaded_encoders_can_transform(self, sample_data, temp_output_dir):
        """Test that loaded encoders can transform data."""
        X, y, encoders = preprocess_data(sample_data)
        
        # Save and load encoders
        joblib.dump(encoders, temp_output_dir / 'encoders.joblib')
        loaded_encoders = joblib.load(temp_output_dir / 'encoders.joblib')
        
        # Test transformation
        test_value = ['Salaried']
        encoded = loaded_encoders['EmploymentType'].transform(test_value)
        assert isinstance(encoded[0], (int, np.integer))


class TestIntegration:
    """Integration tests for the entire pipeline."""
    
    def test_full_pipeline(self, sample_csv_path, temp_output_dir):
        """Test the complete training pipeline."""
        from model import train
        
        # Run training (this will create MLflow runs)
        train(sample_csv_path)
        
        # Check that all expected files are created
        assert (temp_output_dir / 'model1.joblib').exists()
        assert (temp_output_dir / 'model2.joblib').exists()
        assert (temp_output_dir / 'encoders.joblib').exists()
    
    def test_models_can_load_and_predict(self, sample_csv_path, temp_output_dir, sample_data):
        """Test that saved models can be loaded and make predictions."""
        from model import train
        
        # Run training
        train(sample_csv_path)
        
        # Load models
        model1 = joblib.load(temp_output_dir / 'model1.joblib')
        model2 = joblib.load(temp_output_dir / 'model2.joblib')
        encoders = joblib.load(temp_output_dir / 'encoders.joblib')
        
        # Preprocess a sample
        X, y, _ = preprocess_data(sample_data)
        
        # Make predictions
        pred1 = model1.predict(X[:1])
        pred2 = model2.predict(X[:1])
        
        assert len(pred1) == 1
        assert len(pred2) == 1
        assert pred1[0] in ['Low Risk', 'Medium Risk', 'High Risk']
        assert pred2[0] in ['Low Risk', 'Medium Risk', 'High Risk']
