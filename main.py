import os
import csv
import json
import config
from prompts import get_batch_prompt_string
from evaluation import run_evaluation
from request_generator import generate_and_save_response
from data_utils import load_mrc, save_json

CSV_HEADERS = [
    "Shots", "Difficult Examples", "Model Name", "Style", "Total Texts", "Texts Per Batch",
    "Post-Verification", "Accuracy", "Precision", "Recall", "F1", "Token Usage"
]

def save_experiment_to_csv(experiment_data, csv_filepath):
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        row_to_write = {header: experiment_data.get(header) for header in CSV_HEADERS}
        writer.writerow(row_to_write)
    print(f"Результаты эксперимента сохранены в {csv_filepath}")

def run_batch_experiment():
    num_shots = config.BASE_EXPERIMENT_PARAMS["shots"]
    model_name = config.BASE_EXPERIMENT_PARAMS["model"]
    prompt_style = config.BASE_EXPERIMENT_PARAMS["prompt_style"]
    post_verification = config.BASE_EXPERIMENT_PARAMS["post_verification"]
    use_difficult_examples = config.BASE_EXPERIMENT_PARAMS["use_difficult_examples"]
    
    # Загружаем все тексты для разметки
    all_texts = load_mrc(config.TEST_FILE_PATH)
    total_texts = config.BASE_EXPERIMENT_PARAMS["total_texts"]
    batch_size = config.BASE_EXPERIMENT_PARAMS["batch_size"]
    texts_to_annotate = [item["text"] for item in all_texts[:total_texts]]
    
    # Разбиваем на батчи
    batches = [texts_to_annotate[i:i + batch_size] 
               for i in range(0, len(texts_to_annotate), batch_size)]
    
    print(f"Обработка {len(texts_to_annotate)} текстов батчами по {batch_size}")
    print(f"Всего батчей: {len(batches)}")
    
    all_predictions = []
    total_token_usage = 0
    
    # Обрабатываем каждый батч
    for batch_idx, batch_texts in enumerate(batches):
        print(f"Обработка батча {batch_idx + 1}/{len(batches)}...")
        
        prompt_text = get_batch_prompt_string(num_shots, batch_texts, use_difficult_examples)
        batch_file_path = config.DATA_DIR / f"batch_{batch_idx + 1}.json"
        
        token_usage = generate_and_save_response(prompt_text, model_name, batch_file_path)
        total_token_usage += token_usage
        
        # Загружаем результат батча и добавляем к общим предсказаниям
        batch_predictions = load_mrc(batch_file_path)
        all_predictions.extend(batch_predictions)
    
    # Сохраняем объединенные результаты
    final_pred_file_path = config.DATA_DIR / "response.json"
    save_json(all_predictions, final_pred_file_path)
    print(f"Все предсказания объединены в {final_pred_file_path}")
    
    # Вычисляем метрики по всем текстам
    gold_file_path = config.DATA_DIR / "wikiann_100.json"
    metrics = run_evaluation(gold_file_path, final_pred_file_path)
    
    # total_texts и batch_size уже определены выше
    
    actual_metrics = metrics if metrics else {}

    experiment_data_to_save = {
        "Shots": num_shots,
        "Difficult Examples": "Да" if use_difficult_examples else "Нет",
        "Model Name": model_name,
        "Total Texts": total_texts,
        "Texts Per Batch": batch_size,
        "Style": prompt_style,
        "Post-Verification": str(post_verification),
        "Accuracy": actual_metrics.get("accuracy"),
        "Precision": actual_metrics.get("precision"),
        "Recall": actual_metrics.get("recall"),
        "F1": actual_metrics.get("f1"),
        "Token Usage": total_token_usage
    }

    save_experiment_to_csv(experiment_data_to_save, config.CSV_RESULTS_PATH)

if __name__ == "__main__":
    run_batch_experiment()
