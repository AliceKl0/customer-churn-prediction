"""
Модуль исследовательского анализа данных (EDA)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def exploratory_data_analysis(df, output_dir='.'):
    """
    Проводит разведочный анализ данных и создает визуализации.
    
    Анализирует распределения признаков, корреляции между переменными,
    зависимость оттока от различных факторов. Сохраняет два файла:
    - eda_visualizations.png: основные графики распределений и зависимостей
    - correlation_matrix.png: матрица корреляций между числовыми признаками
    
    Args:
        df: Датасет с данными о клиентах (должен содержать колонку 'exited')
        output_dir: Папка для сохранения графиков (по умолчанию текущая)
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Базовая информация
    print("\n1. Информация о датасете:")
    print(f"   Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"\n2. Первые 5 строк:")
    print(df.head())
    
    print("\n3. Статистика:")
    print(df.describe())
    
    print("\n4. Пропущенные значения:")
    print(df.isnull().sum())
    
    print("\n5. Распределение целевой переменной:")
    print(df['exited'].value_counts())
    print(f"   Процент оттока: {df['exited'].mean() * 100:.2f}%")
    
    # Визуализации
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exploratory Data Analysis - Customer Churn', fontsize=16, fontweight='bold')
    
    # Распределение оттока
    df['exited'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Распределение оттока клиентов')
    axes[0, 0].set_xlabel('Отток (0=Нет, 1=Да)')
    axes[0, 0].set_ylabel('Количество')
    axes[0, 0].set_xticklabels(['Остались', 'Ушли'], rotation=0)
    
    # Возраст vs Отток
    sns.boxplot(data=df, x='exited', y='age', ax=axes[0, 1], palette=['#2ecc71', '#e74c3c'])
    axes[0, 1].set_title('Возраст и отток клиентов')
    axes[0, 1].set_xlabel('Отток')
    axes[0, 1].set_xticklabels(['Остались', 'Ушли'])
    
    # Кредитный рейтинг vs Отток
    sns.boxplot(data=df, x='exited', y='credit_score', ax=axes[0, 2], palette=['#2ecc71', '#e74c3c'])
    axes[0, 2].set_title('Кредитный рейтинг и отток')
    axes[0, 2].set_xlabel('Отток')
    axes[0, 2].set_xticklabels(['Остались', 'Ушли'])
    
    # Баланс vs Отток
    sns.boxplot(data=df, x='exited', y='balance', ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
    axes[1, 0].set_title('Баланс счета и отток')
    axes[1, 0].set_xlabel('Отток')
    axes[1, 0].set_xticklabels(['Остались', 'Ушли'])
    
    # География и отток
    geography_churn = pd.crosstab(df['geography'], df['exited'], normalize='index') * 100
    geography_churn.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
    axes[1, 1].set_title('Отток по странам (%)')
    axes[1, 1].set_xlabel('Страна')
    axes[1, 1].set_ylabel('Процент')
    axes[1, 1].legend(['Остались', 'Ушли'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Активность и отток
    activity_churn = pd.crosstab(df['is_active_member'], df['exited'], normalize='index') * 100
    activity_churn.plot(kind='bar', ax=axes[1, 2], color=['#2ecc71', '#e74c3c'])
    axes[1, 2].set_title('Отток по активности (%)')
    axes[1, 2].set_xlabel('Активный член (0=Нет, 1=Да)')
    axes[1, 2].set_ylabel('Процент')
    axes[1, 2].legend(['Остались', 'Ушли'])
    axes[1, 2].set_xticklabels(['Неактивен', 'Активен'], rotation=0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'eda_visualizations.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Визуализации сохранены в '{output_path}'")
    plt.close()
    
    # Корреляционная матрица
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Корреляционная матрица сохранена в '{output_path}'")
    plt.close()

