from data_utils import load_mrc
import config

def extract_entities(json_data):
    """Извлекает сущности из списка JSON-объектов."""
    all_extracted_entities = []
    for item in json_data:
        entities_set = set((e["text"], e["label"]) for e in item.get("entities", []))
        all_extracted_entities.append(entities_set)
    return all_extracted_entities

def calculate_ner_metrics(gold_entities_list, pred_entities_list):
    """Вычисляет метрики TP, FP, FN, Precision, Recall, F1, Accuracy для NER."""
    tp = fp = fn = 0
    for gold_set, pred_set in zip(gold_entities_list, pred_entities_list):
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # Accuracy для NER задач часто определяется иначе, но оставим как в оригинале
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0 

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def run_evaluation(gold_file_path, pred_file_path):
    """Загружает данные, выполняет оценку и выводит метрики."""
    gold_data = load_mrc(gold_file_path)
    pred_data = load_mrc(pred_file_path)

    # Сравниваем только первые N примеров (как в prompts.py и оригинальном evaluation.py)
    n_examples = len(pred_data)
    gold_data_aligned = gold_data[:n_examples]

    gold_entities = extract_entities(gold_data_aligned)
    pred_entities = extract_entities(pred_data)

    metrics = calculate_ner_metrics(gold_entities, pred_entities)

    print(f"Evaluation results for {n_examples} examples:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f} (Note: Accuracy definition might vary for NER)")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

if __name__ == "__main__":
    # Пути к файлам должны соответствовать тем, что используются/генерируются в проекте
    # Например, gold_standard может быть wikiann_100.json, а predictions - response.json
    default_gold_path = config.DATA_DIR / "wikiann_100.json" 
    default_pred_path = config.DATA_DIR / "response.json"
    
    print(f"Running evaluation with Gold: {default_gold_path} and Pred: {default_pred_path}")
    run_evaluation(default_gold_path, default_pred_path)