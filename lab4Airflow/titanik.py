import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import joblib



def download_data():
    try:
        url='https://raw.githubusercontent.com/Scom19/MLOPS_course/main/Titanic.xls'
        df = pd.read_csv(url)
        df.to_csv('titanic.csv', index=False)
        return df
    
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        raise


def clear_data(**kwargs):
    df = pd.read_csv('titanic.csv')

    # Удаление столбцов
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Обработка пропусков
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Преобразование категориальных признаков
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    df.to_csv("cleaned_titanic.csv", index=False)
    return df.to_dict()


def train_model(**kwargs):
    df = pd.read_csv('cleaned_titanic.csv')

    # Разделение данных
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Настройка MLflow
    mlflow.set_experiment("Titanic_Survival_Prediction")
    
    with mlflow.start_run():
        # Создание и обучение пайплайна
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ))
        ])
        
        pipeline.fit(X_train, y_train)

        # Предсказания и метрики
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Логирование параметров
        mlflow.log_params({
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'test_size': 0.2
        })

        # Логирование метрик
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        })

        # Логирование модели
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="titanic_model",
            signature=signature,
            registered_model_name="titanic_rf_model"
        )


        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

# Настройка DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

with DAG(
        'titanic_ml_pipeline1',
        default_args=default_args,
        description='Полный пайплайн обработки Titanic',
        schedule='@weekly',
        catchup=False,
        max_active_runs=1
) as dag:
    download_task = PythonOperator(
        task_id='download_titanic_data',
        python_callable=download_data,
    )

    clean_task = PythonOperator(
        task_id='clean_titanic_data',
        python_callable=clear_data,
    )

    train_task = PythonOperator(
        task_id='train_titanic_model',
        python_callable=train_model,
    )

    download_task >> clean_task >> train_task
