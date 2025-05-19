import pandas as pd

def clean_data():
    df = pd.read_csv('data/raw/titanic.csv')
    
    # Извлечение Title из Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Группировка редких титулов в категорию "Other"
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(4)  # 4 для "Other"
    
    # Удаление ненужных столбцов
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Обработка пропусков
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Кодирование категориальных признаков
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Добавление FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Сохранение очищенных данных
    df.to_csv('data/processed/cleaned_titanic.csv', index=False)
    return df.to_dict()

if __name__ == "__main__":
    clean_data()