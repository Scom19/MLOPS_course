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
    """Загрузка данных Titanic в Airflow"""
    try:
        # Определяем путь к файлу внутри контейнера Airflow
        dag_dir = Path(__file__).parent.absolute()  # Директория DAG
        data_path = dag_dir / "titanic.xls"  # Путь к файлу
        
        # Проверка существования файла
        if not data_path.exists():
            raise FileNotFoundError(f"Файл {data_path} не найден")
        
        # Чтение данных (для Excel используем openpyxl)
        df = pd.read_excel(data_path, engine='openpyxl')
        
        # Сохранение в data-директорию Airflow
        os.makedirs("/opt/airflow/data", exist_ok=True)
        output_path = "/opt/airflow/data/raw_titanic.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Данные успешно загружены. Размер: {df.shape}")
        return df.to_dict()
    
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
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


import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='clean_titanic_data')
    df = pd.DataFrame(data_dict)

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
        f1 = f1_score(y_test, y_pred)

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
            'recall': recall,
            'f1_score': f1
        })

        # Логирование модели
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="titanic_model",
            signature=signature,
            registered_model_name="titanic_rf_model"
        )

        # Сохранение модели локально (дополнительно)
        model_path = "/opt/airflow/models/titanic_model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        print(f"Модель сохранена. Метрики: Accuracy={accuracy:.2f}, Precision={precision:.2f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
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
        'titanic_ml_pipeline',
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
