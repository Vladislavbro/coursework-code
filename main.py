import os
import csv
import json
import config
from data.prompts import get_prompt_string
from evaluation import run_evaluation
from prompt_generator import generate_and_save_response

CSV_HEADERS = [
    "Shots", "Model Name", "Style", "Post-Verification", 
    "Accuracy", "Precision", "Recall", "F1", "Token Usage"
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

def run_single_experiment():

    num_shots = config.BASE_EXPERIMENT_PARAMS["shots"]
    model_name = config.BASE_EXPERIMENT_PARAMS["model"]
    prompt_style = config.BASE_EXPERIMENT_PARAMS["prompt_style"]
    post_verification = config.BASE_EXPERIMENT_PARAMS["post_verification"]

    prompt_text = get_prompt_string(num_shots=num_shots)

    print(f"Отправка запроса к LLM ({model_name})...")
    pred_file_path = config.DATA_DIR / "response.json"
    token_usage = generate_and_save_response(prompt_text, model_name, pred_file_path)

    gold_file_path = config.DATA_DIR / "wikiann_100.json" 
    metrics = run_evaluation(gold_file_path, pred_file_path)
    
    actual_metrics = metrics if metrics else {}

    experiment_data_to_save = {
        "Shots": num_shots,
        "Model Name": model_name,
        "Style": prompt_style,
        "Post-Verification": str(post_verification),
        "Accuracy": actual_metrics.get("accuracy"),
        "Precision": actual_metrics.get("precision"),
        "Recall": actual_metrics.get("recall"),
        "F1": actual_metrics.get("f1"),
        "Token Usage": token_usage 
    }

    save_experiment_to_csv(experiment_data_to_save, config.CSV_RESULTS_PATH)

if __name__ == "__main__":
    run_single_experiment()
