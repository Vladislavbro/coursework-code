import google.generativeai as genai
import json
import config
from data.prompts import prompt  # Готовый промпт из data/prompts.py
from data_utils import save_json # Используем существующую функцию для сохранения JSON

def generate_gemini_response(prompt_text: str, model_name: str, api_key: str):
    """
    Отправляет промпт модели Gemini, получает ответ, очищает его и парсит JSON.
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY не найден. Проверьте файл config.py или переменные окружения.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(prompt_text)
    raw_response = response.text

    # Удаляем markdown-обертку, если она есть (```json ... ```)
    cleaned_response = raw_response
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.strip("```json\n").strip("\n```").strip()
    
    return json.loads(cleaned_response)

def main():
    """
    Основная функция: генерирует ответ от LLM на основе глобального промпта
    и сохраняет его в файл.
    """
    model_name_to_use = config.DEFAULT_MODEL_NAME
    api_key_to_use = config.GEMINI_API_KEY
    output_path = config.DATA_DIR / "response.json"

    print(f"Генерация ответа с использованием модели: {model_name_to_use}...")
    
    gemini_data = generate_gemini_response(
        prompt_text=prompt,
        model_name=model_name_to_use,
        api_key=api_key_to_use
    )
    
    save_json(gemini_data, output_path)
    print(f"Ответ успешно сохранен в {output_path}")

if __name__ == "__main__":
    main()

# Загружаем эталонные аннотации и предсказания Gemini
with open("data/wikiann_100.json", "r", encoding="utf-8") as f:
    gold = json.load(f)
with open("data/response.json", "r", encoding="utf-8") as f:
    pred = json.load(f)

