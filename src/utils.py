"""
Утилиты и вспомогательные функции
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля для визуализаций
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def feature_engineering(df):
    """
    Создает дополнительные признаки на основе существующих.
    
    Добавляет возрастные группы, финансовые индикаторы (баланс/зарплата),
    признаки риска, взаимодействия между переменными и нормализованные значения.
    Эти признаки помогают моделям лучше улавливать нелинейные зависимости.
    
    Args:
        df: Исходный датасет с базовыми признаками
    
    Returns:
        Датасет с добавленными признаками (age_group_encoded, balance_salary_ratio,
        high_risk, age_credit_interaction, total_activity, balance_normalized)
    """
    df_fe = df.copy()
    
    # Возрастные группы
    df_fe['age_group'] = pd.cut(
        df_fe['age'], 
        bins=[0, 30, 45, 60, 100], 
        labels=['18-30', '31-45', '46-60', '60+']
    )
    df_fe['age_group_encoded'] = LabelEncoder().fit_transform(df_fe['age_group'].astype(str))
    
    # Отношение баланса к зарплате (показатель финансовой стабильности)
    df_fe['balance_salary_ratio'] = df_fe['balance'] / (df_fe['estimated_salary'] + 1)
    
    # Признак "рисковый клиент" (низкий кредитный рейтинг + низкий баланс)
    df_fe['high_risk'] = ((df_fe['credit_score'] < 500) & 
                          (df_fe['balance'] < 10000)).astype(int)
    
    # Взаимодействие: возраст и кредитный рейтинг
    df_fe['age_credit_interaction'] = df_fe['age'] * df_fe['credit_score'] / 1000
    
    # Общая активность клиента (комбинация признаков)
    df_fe['total_activity'] = (
        df_fe['num_of_products'] + 
        df_fe['has_cr_card'] + 
        df_fe['is_active_member']
    )
    
    # Нормализованный баланс (относительно среднего)
    mean_balance = df_fe['balance'].mean()
    df_fe['balance_normalized'] = (df_fe['balance'] - mean_balance) / (df_fe['balance'].std() + 1)
    
    # Удаляем временные категориальные признаки
    if 'age_group' in df_fe.columns:
        df_fe = df_fe.drop('age_group', axis=1)
    
    print(f"✓ Feature engineering: добавлено {len(df_fe.columns) - len(df.columns)} новых признаков")
    
    return df_fe

def handle_imbalance(X_train, y_train, method='smote'):
    """
    Балансирует классы в обучающей выборке с помощью SMOTE.
    
    SMOTE создает синтетические примеры минорного класса, чтобы выровнять
    соотношение классов. Это особенно важно при работе с несбалансированными
    данными, где один класс значительно преобладает над другим.
    
    Args:
        X_train: Матрица признаков обучающей выборки
        y_train: Вектор целевой переменной обучающей выборки
        method: Метод балансировки (поддерживается только 'smote')
    
    Returns:
        Кортеж (X_resampled, y_resampled) с сбалансированными данными.
        Если SMOTE недоступен, возвращает исходные данные без изменений.
    """
    try:
        from imblearn.over_sampling import SMOTE
        SMOTE_AVAILABLE = True
    except ImportError:
        SMOTE_AVAILABLE = False
        print("  ⚠ SMOTE недоступен, пропускаем обработку дисбаланса")
        return X_train, y_train
    
    if not SMOTE_AVAILABLE:
        return X_train, y_train
    
    print(f"\n  Обработка дисбаланса классов (метод: {method})...")
    print(f"  До: {pd.Series(y_train).value_counts().to_dict()}")
    
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    else:
        return X_train, y_train
    
    print(f"  После: {pd.Series(y_res).value_counts().to_dict()}")
    
    return X_res, y_res

