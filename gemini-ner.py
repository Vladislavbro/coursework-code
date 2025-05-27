import google.generativeai as genai
import json
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from data.prompts import prompt, texts_to_annotate


API_KEY = "AIzaSyDqwBO7fYRUtmWktEXnXTzn-RX67zO2Pi4"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemma-3n-e4b-it')

# 1. Получаем ответ от Gemini
response = model.generate_content(prompt)
raw_response = response.text

# Удаляем markdown-обертку, если она есть (```json ... ```)
cleaned_response = raw_response
if cleaned_response.startswith("```json"):
    cleaned_response = cleaned_response.strip("```json\n")
    cleaned_response = cleaned_response.strip("\n```")
    cleaned_response = cleaned_response.strip()

# Парсим очищенный JSON-текст
gemini_data = json.loads(cleaned_response)

# Сохраняем обработанный ответ Gemini в data/response.json
with open("data/response.json", "w", encoding="utf-8") as f:
    json.dump(gemini_data, f, ensure_ascii=False, indent=2)

# Загружаем эталонные аннотации и предсказания Gemini
with open("data/wikiann_100.json", "r", encoding="utf-8") as f:
    gold = json.load(f)
with open("data/response.json", "r", encoding="utf-8") as f:
    pred = json.load(f)

# Сравниваем только первые N примеров (как в prompts.py)
N = len(pred)
gold = gold[:N]

def extract_entities(json_data):
    all_entities = []
    for item in json_data:
        entities = set((e["text"], e["label"]) for e in item.get("entities", []))
        all_entities.append(entities)
    return all_entities

gold_entities = extract_entities(gold)
pred_entities = extract_entities(pred)

TP = FP = FN = 0
for gold_set, pred_set in zip(gold_entities, pred_entities):
    TP += len(gold_set & pred_set)
    FP += len(pred_set - gold_set)
    FN += len(gold_set - pred_set)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1:        {f1:.3f}")
print(f"Accuracy:  {accuracy:.3f}")