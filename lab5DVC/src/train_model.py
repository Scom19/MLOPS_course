import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import mlflow
from mlflow.models import infer_signature
import json
import os

def train_model():
    df = pd.read_csv('data/processed/cleaned_titanic.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save test set for the test_model stage
    test_df = pd.concat([X_test, y_test], axis=1)
    os.makedirs('data/processed', exist_ok=True)
    test_df.to_csv('data/processed/test_titanic.csv', index=False)

    mlflow.set_experiment("Titanic_Survival_Prediction")
    with mlflow.start_run():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        mlflow.log_metrics(metrics)
        
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/metrics_train.json', 'w') as f:
            json.dump(metrics, f)
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, 'models/titanic_model.pkl')
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, "titanic_model", registered_model_name="titanic_rf_model")

    return metrics

if __name__ == "__main__":
    train_model()