"""
Customer Churn Prediction - ML Project
Главный файл для запуска анализа оттока клиентов
"""

from src.data_loader import load_data
from src.eda import exploratory_data_analysis
from src.model_training import prepare_data, train_models, feature_importance_analysis

def main():
    """Основная функция"""
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION - ML PROJECT")
    print("="*60 + "\n")
    
    # Загрузка данных
    print("Загрузка данных...")
    df = load_data()
    
    # EDA
    exploratory_data_analysis(df, output_dir='results')
    
    # Подготовка данных
    print("\nПодготовка данных для моделирования...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df, handle_imbalance_flag=True)
    
    # Обучение моделей
    results, best_model_name = train_models(X_train, X_test, y_train, y_test, output_dir='results')
    
    # Анализ важности признаков для Random Forest или XGBoost
    tree_models = ['Random Forest', 'XGBoost', 'Random Forest (Tuned)', 'XGBoost (Tuned)']
    for model_name in tree_models:
        if model_name in results:
            feature_importance_analysis(results[model_name]['model'], feature_names, output_dir='results')
            break
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("="*60)
    print("\nСозданные файлы:")
    print("  - results/eda_visualizations.png")
    print("  - results/correlation_matrix.png")
    print("  - results/model_performance.png")
    print("  - results/roc_curves.png")
    print("  - results/feature_importance.png")
    print("  - models/best_model.pkl")

if __name__ == "__main__":
    main()
