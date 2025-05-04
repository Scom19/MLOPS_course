import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def download_data():
    url = "https://github.com/Scom19/MLOPS_course/raw/main/Titanic.xls"

    try:
        df = pd.read_excel(url, engine='openpyxl')
        df.to_csv("/opt/airflow/data/raw_titanic.csv", index=False)
        print(f"Данные загружены. Размер: {df.shape}")
        return df.to_dict()
    except Exception as e:
        print(f"Ошибка загрузки: {str(e)}")
        raise


def clear_data(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='download_titanic_data')
    df = pd.DataFrame(data_dict)

    # Удаление столбцов
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Обработка пропусков
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Преобразование категориальных признаков
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    df.to_csv("/opt/airflow/data/cleaned_titanic.csv", index=False)
    return df.to_dict()


def train_model(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='clean_titanic_data')
    df = pd.DataFrame(data_dict)

    # Разделение данных
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создание пайплайна
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])

    # Обучение модели
    pipeline.fit(X_train, y_train)

    # Предсказания и метрики
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Сохранение модели
    joblib.dump(pipeline, '/opt/airflow/models/titanic_model.pkl')

    # Сохранение метрик
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    return metrics

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
        'titanic_ml_pipeline',
        default_args=default_args,
        description='Полный пайплайн обработки Titanic',
        schedule_interval='@weekly',
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
