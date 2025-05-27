from pathlib import Path
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Директории проекта ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# --- Пути к файлам данных ---
TRAIN_FILE_PATH = DATA_DIR / "wikiann_18.json"
TEST_FILE_PATH = DATA_DIR / "wikiann_100.json"
PROMPTS_FILE_PATH = DATA_DIR / "prompts.py"

# --- Пути к файлам результатов ---
CSV_RESULTS_PATH = DATA_DIR / "results.csv"

# --- Общие настройки экспериментов ---
DEFAULT_MODEL_NAME = "gemma-3n-e4b-it"
AVAILABLE_MODELS = [
    "gemma-3n-e4b-it",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash"
]

# --- Базовые параметры для одного запуска эксперимента ---
BASE_EXPERIMENT_PARAMS = {
    "model": DEFAULT_MODEL_NAME,
    "shots": 0,
    "example_selection_mode": "random",
    "prompt_style": "json-schema",
    "post_verification": False,
}