import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
import numpy as np

# Set experiment name
mlflow.set_experiment("Loan Risk Prediction")


def load_data(file_path):
    """
    Load loan risk data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocess the dataset by encoding categorical variables.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X, y, encoders) - Features, target, and label encoders
    """
    data = df.copy()
    categorical_cols = ['EmploymentType', 'ResidenceType', 'PreviousDefault']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    
    X = data.drop('RiskCategory', axis=1)
    y = data['RiskCategory']
    
    return X, y, encoders


def log_model_metrics(y_true, y_pred, model_name, classes):
    """
    Log comprehensive metrics to MLflow.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model for logging
        classes: List of class labels
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    mlflow.log_metric(f"{model_name}_accuracy", accuracy)
    mlflow.log_metric(f"{model_name}_precision_weighted", precision)
    mlflow.log_metric(f"{model_name}_recall_weighted", recall)
    mlflow.log_metric(f"{model_name}_f1_weighted", f1)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=classes)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=classes)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=classes)
    
    for i, class_name in enumerate(classes):
        mlflow.log_metric(f"{model_name}_precision_{class_name.replace(' ', '_')}", precision_per_class[i])
        mlflow.log_metric(f"{model_name}_recall_{class_name.replace(' ', '_')}", recall_per_class[i])
        mlflow.log_metric(f"{model_name}_f1_{class_name.replace(' ', '_')}", f1_per_class[i])
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Save confusion matrix as text artifact
    cm_str = f"Confusion Matrix for {model_name}:\n"
    cm_str += f"Classes: {classes}\n\n"
    cm_str += str(cm)
    
    with open(f"{model_name}_confusion_matrix.txt", "w") as f:
        f.write(cm_str)
    mlflow.log_artifact(f"{model_name}_confusion_matrix.txt")
    
    # Save classification report
    report = classification_report(y_true, y_pred)
    with open(f"{model_name}_classification_report.txt", "w") as f:
        f.write(f"Classification Report for {model_name}:\n\n")
        f.write(report)
    mlflow.log_artifact(f"{model_name}_classification_report.txt")
    
    return accuracy, precision, recall, f1


def train_random_forest(X_train, y_train, X_test, y_test, classes):
    """
    Train Random Forest model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classes: List of class labels
        
    Returns:
        RandomForestClassifier: Trained model
    """
    with mlflow.start_run(run_name="Random_Forest_Classifier"):
        # Model parameters
        params = {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Log metrics
        accuracy, precision, recall, f1 = log_model_metrics(
            y_test, y_pred, "RandomForest", classes
        )
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("rf_feature_importance.csv", index=False)
        mlflow.log_artifact("rf_feature_importance.csv")
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Save model locally
        joblib.dump(model, 'model1.joblib')
        mlflow.log_artifact('model1.joblib')
        
        print(f"\n{'='*60}")
        print(f"Random Forest Classifier Results")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"{'='*60}\n")
        
        return model


def train_logistic_regression(X_train, y_train, X_test, y_test, classes):
    """
    Train Logistic Regression model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classes: List of class labels
        
    Returns:
        LogisticRegression: Trained model
    """
    with mlflow.start_run(run_name="Logistic_Regression"):
        # Model parameters
        params = {
            "solver": "lbfgs",
            "max_iter": 1000,
            "multi_class": "auto",
            "random_state": 8888
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Log metrics
        accuracy, precision, recall, f1 = log_model_metrics(
            y_test, y_pred, "LogisticRegression", classes
        )
        
        # Log model
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        
        # Save model locally
        joblib.dump(model, 'model2.joblib')
        mlflow.log_artifact('model2.joblib')
        
        print(f"\n{'='*60}")
        print(f"Logistic Regression Results")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"{'='*60}\n")
        
        return model


def train(file_path):
    """
    Main training pipeline that trains both models and compares performance.
    
    Args:
        file_path (str): Path to the training data CSV file
    """
    print("\n" + "="*60)
    print("Starting Loan Risk Prediction Model Training")
    print("="*60 + "\n")
    
    # Load and preprocess data
    df = load_data(file_path)
    X, y, encoders = preprocess_data(df)
    
    # Get unique classes
    classes = sorted(y.unique())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Classes: {classes}\n")
    
    # Train Random Forest
    model1 = train_random_forest(X_train, y_train, X_test, y_test, classes)
    
    # Train Logistic Regression
    model2 = train_logistic_regression(X_train, y_train, X_test, y_test, classes)
    
    # Save encoders
    joblib.dump(encoders, 'encoders.joblib')
    
    # Log encoders as artifact in a separate run
    with mlflow.start_run(run_name="Data_Preprocessing"):
        mlflow.log_artifact('encoders.joblib')
        mlflow.log_param("categorical_columns", str(['EmploymentType', 'ResidenceType', 'PreviousDefault']))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Models saved:")
    print("  - model1.joblib (Random Forest)")
    print("  - model2.joblib (Logistic Regression)")
    print("  - encoders.joblib (Label Encoders)")
    print("\nView results in MLflow UI:")
    print("  Run: mlflow ui")
    print("  Then open: http://localhost:5000")
    print("="*60 + "\n")


if __name__ == "__main__":
    train('../data/loan_risk_data.csv')