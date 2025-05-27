import google.generativeai as genai
import json
import config
from data_utils import save_json

def generate_gemini_response(prompt_text: str, model_name: str, api_key: str):
    """
    Отправляет промпт модели Gemini, получает ответ, очищает его и парсит JSON.
    """

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(prompt_text)
    raw_response = response.text

    # Удаляем markdown-обертку, если она есть (```json ... ```)
    cleaned_response = raw_response
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.strip("```json\n").strip("\n```").strip()
    
    token_count = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
    return json.loads(cleaned_response), token_count

def generate_and_save_response(prompt_text: str, model_name: str, output_path: str):
    api_key = config.GEMINI_API_KEY
    gemini_data, token_usage = generate_gemini_response(prompt_text, model_name, api_key)
    save_json(gemini_data, output_path)
    return token_usage

if __name__ == "__main__":
    print("Этот модуль предназначен для импорта, а не для прямого запуска.")

