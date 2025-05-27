import config
import itertools
import csv
import os

# Импорты для будущих вызовов (пока не используются в базовой структуре)
# import prompt_generator 
# import evaluation
# import data_utils # для логирования результатов

# Колонки для CSV (согласно PROJECT_STRUCTURE.md и выводу evaluation.py)
CSV_HEADERS = [
    "Shots", "Model Name", "Style", "Post-Verification", 
    "Accuracy", "Precision", "Recall", "F1", "Token Usage",
    "TP", "FP", "FN" 
    # Добавь сюда "Variance F1", "Self-verification Gain" если они будут
]

def save_experiment_to_csv(experiment_data, csv_filepath):
    """Сохраняет данные одного эксперимента в CSV файл."""
    # Убедимся, что директория для результатов существует
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    
    file_exists = os.path.isfile(csv_filepath)
    try:
        with open(csv_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()  # Записываем заголовки, если файл новый
            
            # Готовим данные для записи, убеждаясь, что все ключи из CSV_HEADERS присутствуют
            row_to_write = {header: experiment_data.get(header) for header in CSV_HEADERS}
            writer.writerow(row_to_write)
    except IOError as e:
        print(f"Ошибка при записи в CSV файл {csv_filepath}: {e}")


def run_all_experiments():
    """
    Итерируется по всем наборам экспериментов, определенным в config.EXPERIMENT_SUITES,
    и для каждой комбинации параметров выводит информацию о планируемом запуске.
    """
    for suite_name, suite_config in config.EXPERIMENT_SUITES.items():
        print(f"\n--- Запуск серии экспериментов: {suite_name} ---")
        print(f"Описание: {suite_config['description']}")

        base_params = suite_config['base_params']
        varying_params_config = suite_config['varying_params']

        if not varying_params_config:
            print(f"  Внимание: для серии {suite_name} не найдены варьируемые параметры. Запуск только с базовыми параметрами.")
            current_run_params = base_params.copy()
            print(f"    Планируемый запуск с параметрами: {current_run_params}")
            # TODO: Здесь будет вызов генерации промпта, LLM, оценки и логирования
            continue

        param_names = list(varying_params_config.keys())
        value_lists = [varying_params_config[name] for name in param_names]

        for value_combination in itertools.product(*value_lists):
            current_run_params = base_params.copy()
            for i, param_name in enumerate(param_names):
                current_run_params[param_name] = value_combination[i]
            
            print(f"    Планируемый запуск с параметрами: {current_run_params}")
            # TODO: На этом месте будут следующие шаги:
            # 1. Динамическая подготовка/изменение промпта на основе current_run_params.
            #    (потребует доработки data/prompts.py и, возможно, prompt_generator.py)
            # 2. Вызов функции из prompt_generator.py для получения ответа от LLM.
            #    (например, prompt_generator.generate_gemini_response(dynamic_prompt, model, api_key))
            #    Сохранение ответа в уникальный файл.
            # 3. Вызов функции из evaluation.py для оценки полученного ответа.
            #    (например, evaluation.run_evaluation(gold_file_path, unique_response_file_path))
            #    Эта функция вернет словарь с метриками: {"precision": ..., "recall": ..., "f1": ..., "accuracy": ..., "TP": ..., "FP": ..., "FN": ...}
            # 4. Логирование параметров (current_run_params) и метрик в CSV.
            #    Пример:
            #    metrics = evaluation.run_evaluation(gold_path, pred_path) # Должна вернуть словарь метрик
            #    token_usage = 1234 # Получить реальное значение
            #    
            #    experiment_data_to_save = {
            #        "Shots": current_run_params.get("shots"),
            #        "Model Name": current_run_params.get("model"),
            #        "Style": current_run_params.get("prompt_style"),
            #        "Post-Verification": str(current_run_params.get("post_verification")), # Преобразуем bool в str для CSV
            #        "Accuracy": metrics.get("accuracy"),
            #        "Precision": metrics.get("precision"),
            #        "Recall": metrics.get("recall"),
            #        "F1": metrics.get("f1"),
            #        "TP": metrics.get("TP"),
            #        "FP": metrics.get("FP"),
            #        "FN": metrics.get("FN"),
            #        "Token Usage": token_usage 
            #    }
            #    save_experiment_to_csv(experiment_data_to_save, config.CSV_RESULTS_PATH)
            #    print(f"Результаты для {current_run_params} сохранены.")


def main():
    """
    Основная точка входа для запуска всех экспериментов.
    """
    run_all_experiments()

if __name__ == "__main__":
    main() 