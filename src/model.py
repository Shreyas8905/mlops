import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    data = df.copy()
    categorical_cols = ['EmploymentType', 'ResidenceType', 'PreviousDefault']
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('RiskCategory', axis=1)
    y = data['RiskCategory']
    return X, y

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"--- Model Performance ---")
    print(f"Accuracy Score: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix')
    # plt.savefig('confusion_matrix.png')
    return accuracy

def main(file_path):
    df = load_data(file_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main('../Data/loan_risk_data.csv')
