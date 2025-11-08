"""
Модуль загрузки данных
"""

import pandas as pd
import numpy as np

def load_data(data_path=None):
    """
    Загружает данные о клиентах из CSV файла или создает синтетический датасет.
    
    Сначала пытается загрузить данные по указанному пути, затем проверяет
    стандартную директорию data/raw/. Если файлы не найдены, генерирует
    синтетические данные с реалистичными зависимостями для демонстрации.
    
    Args:
        data_path: Путь к CSV файлу. Если None, используется стандартный путь
                  или генерируются синтетические данные.
    
    Returns:
        DataFrame с колонками: age, credit_score, tenure, balance, 
        num_of_products, has_cr_card, is_active_member, estimated_salary,
        geography, gender, exited (целевая переменная)
    """
    # Попытка загрузить реальные данные
    if data_path:
        try:
            df = pd.read_csv(data_path)
            print(f"✓ Загружен реальный датасет из {data_path}")
            return df
        except FileNotFoundError:
            print(f"⚠ Файл {data_path} не найден, используем синтетические данные...")
    
    # Попытка загрузить из стандартной директории
    try:
        df = pd.read_csv('data/raw/Churn_Modelling.csv')
        print("✓ Загружен реальный датасет из data/raw/Churn_Modelling.csv")
        return df
    except FileNotFoundError:
        pass
    
    # Создаем синтетический датасет для демонстрации
    print("Генерация синтетических данных...")
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'tenure': np.random.randint(0, 10, n_samples),
        'balance': np.random.normal(50000, 20000, n_samples).clip(0),
        'num_of_products': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'has_cr_card': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'is_active_member': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'estimated_salary': np.random.normal(100000, 30000, n_samples).clip(0),
        'geography': np.random.choice(['France', 'Germany', 'Spain'], n_samples, p=[0.5, 0.25, 0.25]),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
    }
    
    df = pd.DataFrame(data)
    
    # Создаем целевую переменную с логикой
    churn_prob = (
        (df['age'] > 60) * 0.3 +
        (df['credit_score'] < 500) * 0.4 +
        (df['balance'] < 10000) * 0.2 +
        (df['num_of_products'] == 1) * 0.15 +
        (df['is_active_member'] == 0) * 0.25 +
        np.random.random(n_samples) * 0.3
    )
    df['exited'] = (churn_prob > 0.5).astype(int)
    
    return df

