import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import mlflow
import json
import os
import numpy as np

def test_model():
    df = pd.read_csv('data/processed/cleaned_titanic.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    model = joblib.load('models/titanic_model.pkl')
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'auc': make_scorer(roc_auc_score)
    }

    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=False)

    metrics = {
        'accuracy_mean': np.mean(cv_results['test_accuracy']),
        'precision_mean': np.mean(cv_results['test_precision']),
        'recall_mean': np.mean(cv_results['test_recall']),
        'auc_mean': np.mean(cv_results['test_auc'])
    }

    mlflow.set_experiment("Titanic_Survival_Prediction")
    with mlflow.start_run():
        mlflow.log_metrics({
            'test_accuracy_mean': metrics['accuracy_mean'],
            'test_precision_mean': metrics['precision_mean'],
            'test_recall_mean': metrics['recall_mean'],
            'test_auc_mean': metrics['auc_mean']
        })

        os.makedirs('metrics', exist_ok=True)
        with open('metrics/metrics_test.json', 'w') as f:
            json.dump(metrics, f)

    return metrics

if __name__ == "__main__":
    test_model()