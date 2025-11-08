"""
Модуль обучения и оценки моделей машинного обучения
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os
import pickle

from src.utils import feature_engineering, handle_imbalance

# Функция для безопасной проверки доступности XGBoost
def _check_xgboost_available():
    """Проверка доступности XGBoost с обработкой всех возможных ошибок"""
    try:
        from xgboost import XGBClassifier
        return True, XGBClassifier
    except (ImportError, Exception) as e:
        return False, None

def prepare_data(df, handle_imbalance_flag=True):
    """
    Подготавливает данные для обучения моделей.
    
    Выполняет полный пайплайн предобработки: создание новых признаков,
    кодирование категориальных переменных, разделение на train/test,
    стандартизацию признаков и опциональную балансировку классов через SMOTE.
    
    Args:
        df: Исходный датасет с данными о клиентах
        handle_imbalance_flag: Если True, применяет SMOTE для балансировки классов
    
    Returns:
        Кортеж из (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
        где scaler - обученный StandardScaler для масштабирования новых данных
    """
    # Применяем feature engineering
    df_processed = feature_engineering(df)
    
    # Кодируем категориальные переменные
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Разделяем признаки и целевую переменную
    X = df_processed.drop('exited', axis=1)
    y = df_processed['exited']
    
    # Сохраняем названия признаков для анализа важности
    feature_names = X.columns.tolist()
    
    # Разделяем на train и test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обработка дисбаланса классов
    if handle_imbalance_flag:
        X_train_scaled, y_train = handle_imbalance(X_train_scaled, y_train)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def train_models(X_train, X_test, y_train, y_test, output_dir='.'):
    """
    Обучает несколько моделей и выбирает лучшую с подбором гиперпараметров.
    
    Тестирует Logistic Regression, Random Forest и XGBoost (если доступен).
    Для лучшей модели выполняет GridSearchCV для оптимизации гиперпараметров.
    Сохраняет лучшую модель в models/best_model.pkl и создает визуализации.
    
    Args:
        X_train: Обучающая выборка признаков (уже масштабированная)
        X_test: Тестовая выборка признаков (уже масштабированная)
        y_train: Обучающая выборка целевой переменной
        y_test: Тестовая выборка целевой переменной
        output_dir: Папка для сохранения графиков и модели
    
    Returns:
        Кортеж (results_dict, best_model_name), где results_dict содержит
        результаты всех моделей с метриками и предсказаниями
    """
    print("\n" + "=" * 60)
    print("MACHINE LEARNING MODELS")
    print("=" * 60)
    
    # Базовые модели
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    # Добавляем XGBoost если доступен
    xgboost_available, XGBClassifier = _check_xgboost_available()
    if xgboost_available:
        models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    else:
        print("\n⚠ XGBoost недоступен (требуется OpenMP runtime). Продолжаем без XGBoost.")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Обучение
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Метрики
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"\nCross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores
        }
    
    # Подбор гиперпараметров для лучшей модели
    print("\n" + "-" * 60)
    print("Подбор гиперпараметров для лучшей модели...")
    best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    print(f"Выбрана модель: {best_model_name}")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif best_model_name == 'XGBoost' and xgboost_available:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        base_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    else:
        print("  ⚠ Пропускаем GridSearch для этой модели")
        param_grid = None
        base_model = None
    
    if param_grid is not None and base_model is not None:
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\nЛучшие параметры: {grid_search.best_params_}")
        print(f"Лучший CV score: {grid_search.best_score_:.4f}")
        
        # Оценка на тестовой выборке
        best_model = grid_search.best_estimator_
        y_pred_tuned = best_model.predict(X_test)
        y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]
        roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
        
        print(f"Test ROC-AUC (tuned): {roc_auc_tuned:.4f}")
        
        results[f'{best_model_name} (Tuned)'] = {
            'model': best_model,
            'y_pred': y_pred_tuned,
            'y_pred_proba': y_pred_proba_tuned,
            'roc_auc': roc_auc_tuned,
            'cv_scores': None
        }
        
        # Сохраняем лучшую модель в корневую папку models
        model_path = os.path.join('models', 'best_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"\n✓ Лучшая модель сохранена в '{model_path}'")
    
    # Визуализация результатов
    visualize_results(results, y_test, output_dir)
    
    # Выбираем лучшую модель
    final_best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    print(f"\n{'='*60}")
    print(f"Лучшая модель: {final_best_model_name}")
    print(f"ROC-AUC: {results[final_best_model_name]['roc_auc']:.4f}")
    print(f"{'='*60}")
    
    return results, final_best_model_name

def visualize_results(results, y_test, output_dir='.'):
    """
    Создает визуализации для сравнения моделей.
    
    Генерирует два графика:
    - model_performance.png: матрицы ошибок для всех моделей
    - roc_curves.png: ROC-кривые для сравнения качества классификации
    
    Args:
        results: Словарь с результатами моделей (ключ - название, значение - dict
                 с ключами 'y_pred', 'y_pred_proba', 'roc_auc')
        y_test: Реальные значения целевой переменной для тестовой выборки
        output_dir: Папка для сохранения графиков
    """
    n_models = len(results)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Confusion Matrices
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Остались', 'Ушли'],
                   yticklabels=['Остались', 'Ушли'])
        axes[idx].set_title(f'{name}\nROC-AUC: {result["roc_auc"]:.4f}')
        axes[idx].set_ylabel('Реальные значения')
        axes[idx].set_xlabel('Предсказанные значения')
    
    # Скрываем лишние subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Результаты моделей сохранены в '{output_path}'")
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC кривые сохранены в '{output_path}'")
    plt.close()

def feature_importance_analysis(model, feature_names, output_dir='.'):
    """
    Анализирует важность признаков для tree-based моделей.
    
    Использует встроенный метод feature_importances_ для Random Forest или XGBoost.
    Создает горизонтальный бар-график с топ-15 признаками и выводит топ-10 в консоль.
    
    Args:
        model: Обученная модель с методом feature_importances_ (Random Forest, XGBoost)
        feature_names: Список названий признаков в том же порядке, что и в данных
        output_dir: Папка для сохранения графика feature_importance.png
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Показываем топ-15 признаков
        top_n = min(15, len(indices))
        
        plt.figure(figsize=(10, 8))
        model_type = type(model).__name__
        plt.title(f'Топ-{top_n} важных признаков ({model_type})', fontsize=14, fontweight='bold')
        plt.barh(range(top_n), importances[indices[:top_n]], color='#3498db')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
        plt.xlabel('Важность', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Анализ важности признаков сохранен в '{output_path}'")
        plt.close()
        
        # Выводим топ-10 в консоль
        print(f"\nТоп-10 важных признаков:")
        for i in range(min(10, len(indices))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

