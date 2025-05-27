from pathlib import Path
import os

# --- Директории проекта ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# --- Пути к файлам данных ---
TRAIN_FILE_PATH = DATA_DIR / "wikiann_train.json"
TEST_FILE_PATH = DATA_DIR / "wikiann_test.json"
PROMPTS_FILE_PATH = DATA_DIR / "prompts.py"

# --- Пути к файлам результатов ---
DB_PATH = RESULTS_DIR / "experiments.db"
CSV_RESULTS_PATH = RESULTS_DIR / "results.csv"  # Альтернативный/дополнительный формат для результатов

# --- Общие настройки экспериментов ---
DEFAULT_MODEL_NAME = "gemma-3n-e4b-it"
AVAILABLE_MODELS = [
    "gemma-3n-e4b-it",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash"
]

# Модель для генерации эмбеддингов (используется в kNN выборе примеров)
EMBEDDING_MODEL = "gemini-embedding-exp-03-07" 

# --- Базовые параметры для одного запуска эксперимента ---
# Эти параметры могут переопределяться для конкретных серий экспериментов.
BASE_EXPERIMENT_PARAMS = {
    "model": DEFAULT_MODEL_NAME,
    "shots": 3,  # Количество примеров в промпте
    "example_selection_mode": "knn",  # Метод выбора примеров: 'random' или 'knn'
    "prompt_style": "json-schema",    # Стиль промпта: 'json-schema' или 'cot' (Chain-of-Thought)
    "post_verification": False,       # Включена ли пост-верификация
}

# --- Конфигурации для серий экспериментов (согласно PROJECT_STRUCTURE.md) ---
# Каждая запись соответствует одному из экспериментов, описанных в PROJECT_STRUCTURE.md.
# `run_experiment.py` может использовать эту структуру для итерации по различным параметрам.
EXPERIMENT_SUITES = {
    "1_number_of_shots": {
        "description": "Влияние количества примеров (shots) на качество.",
        "base_params": {
            **BASE_EXPERIMENT_PARAMS,
            "example_selection_mode": "random",
            "post_verification": False,
            "model": DEFAULT_MODEL_NAME, # Фиксируем модель для этого эксперимента
            "prompt_style": "json-schema" # Фиксируем стиль промпта
        },
        "varying_params": {"shots": [0, 1, 3, 5]}
    },
    "2_example_selection": {
        "description": "Сравнение методов выбора примеров: random vs kNN.",
        "base_params": {
            **BASE_EXPERIMENT_PARAMS,
            "shots": 3,
            "post_verification": False,
            "model": DEFAULT_MODEL_NAME,
            "prompt_style": "json-schema"
        },
        "varying_params": {"example_selection_mode": ["random", "knn"]}
    },
    "3_prompt_style": {
        "description": "Влияние стиля промпта: JSON-schema vs Chain-of-Thought.",
        "base_params": {
            **BASE_EXPERIMENT_PARAMS,
            "shots": 3,
            "example_selection_mode": "knn",
            "post_verification": False,
            "model": DEFAULT_MODEL_NAME
        },
        "varying_params": {"prompt_style": ["json-schema", "cot"]}
    },
    "4_model_variant": {
        "description": "Сравнение различных LLM.",
        "base_params": {
            **BASE_EXPERIMENT_PARAMS,
            "shots": 3,
            "example_selection_mode": "knn",
            "prompt_style": "json-schema",
            "post_verification": False
        },
        "varying_params": {"model": AVAILABLE_MODELS} # Используются все модели из AVAILABLE_MODELS
    },
    "5_post_verification": {
        "description": "Эффект от пост-верификации.",
        "base_params": {
            **BASE_EXPERIMENT_PARAMS,
            "shots": 3,
            "example_selection_mode": "knn",
            "prompt_style": "json-schema",
            "model": DEFAULT_MODEL_NAME
        },
        "varying_params": {"post_verification": [False, True]}
    },
}

# --- Метрики для логирования ---
# Список метрик, которые будут сохраняться в базе данных результатов.
METRICS_TO_LOG = ["accuracy", "precision", "recall", "f1_score", "token_usage_input", "token_usage_output"]

# --- Типы сущностей (NER) ---
# Примерный список типов сущностей, если задача связана с NER (например, для датасета wikiann)
ENTITY_TYPES = ["PER", "LOC", "ORG", "MISC"] # или другие релевантные типы


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")