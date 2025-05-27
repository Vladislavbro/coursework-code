import json
from data_utils import load_mrc
import random
from config import TRAIN_FILE_PATH, DATA_DIR


def get_batch_prompt_string(num_shots, batch_texts, use_difficult_examples=True):
    random.seed(42)
    all_examples = load_mrc(TRAIN_FILE_PATH)
    example_json = random.sample(all_examples, num_shots) if num_shots > 0 else []
    difficult_examples = load_mrc(DATA_DIR / "difficult_examples.json") if use_difficult_examples else []

    prompt = f"""
Тебе будет предоставлен список текстов. Твоя задача — извлечь из каждого текста именованные сущности (людей - PER, организации - ORG, локации - LOC).
Ты ДОЛЖЕН вернуть результат в виде списка JSON-объектов. Каждый JSON-объект в списке должен соответствовать одному входному тексту и иметь два ключа:
1.  `"text"`: оригинальный текст.
2.  `"entities"`: список JSON-объектов, где каждый объект представляет найденную сущность и имеет ключи `"text"` (текст сущности) и `"label"` (тип сущности: PER, ORG или LOC).

ВАЖНО: Твой ответ должен быть СТРОГО в формате JSON, точно как в примере ниже. Не добавляй никаких объяснений, комментариев или какого-либо другого текста вне JSON-структуры.

Пример формата вывода (это ТОЛЬКО пример структуры, не используй эти данные для реальных текстов):
{json.dumps(example_json, ensure_ascii=False, indent=2)}

{f"Обрати внимание на эти сложные примеры правильной разметки:\n{json.dumps(difficult_examples, ensure_ascii=False, indent=2)}\n" if use_difficult_examples and difficult_examples else ""}В тексте всегда есть как минимум 1 тип сущности, если ты его не нашел с первого раза, проверь еще раз.

Тексты для разметки:
{json.dumps(batch_texts, ensure_ascii=False, indent=2)}

Пожалуйста, верни ТОЛЬКО JSON-массив с результатами разметки для КАЖДОГО из предоставленных текстов. Не генерируй Python-код или какой-либо другой текст.
"""
    return prompt