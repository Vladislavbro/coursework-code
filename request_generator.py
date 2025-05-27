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

def main():
    """
    Основная функция: генерирует ответ от LLM на основе глобального промпта
    и сохраняет его в файл.
    """
    from prompts import get_prompt_string
    model_name_to_use = config.DEFAULT_MODEL_NAME
    output_path = config.DATA_DIR / "response.json"

    print(f"Генерация ответа с использованием модели: {model_name_to_use}...")
    
    prompt_text = get_prompt_string(num_shots=config.BASE_EXPERIMENT_PARAMS["shots"])
    token_usage = generate_and_save_response(prompt_text, model_name_to_use, output_path)
    print(f"Ответ успешно сохранен в {output_path}, использовано токенов: {token_usage}")

if __name__ == "__main__":
    main()

