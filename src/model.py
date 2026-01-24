import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
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

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main(file_path):
    df = load_data(file_path)
    X, y, encoders = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = train_random_forest(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'model.joblib')
    joblib.dump(encoders, 'encoders.joblib')
    print("Success: Model and Encoders saved to disk!")

if __name__ == "__main__":
    main('../Data/loan_risk_data.csv')