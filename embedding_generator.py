from sentence_transformers import SentenceTransformer
from data_utils import load_mrc
from config import TRAIN_FILE_PATH

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(data_path=TRAIN_FILE_PATH):
    """
    Загружает тексты из JSON-файла, генерирует для них эмбеддинги и возвращает их.

    Args:
        data_path (str): Путь к JSON-файлу с данными.

    Returns:
        tuple: Кортеж из двух списков:
               - список оригинальных текстов
               - список numpy.ndarray с эмбеддингами для каждого текста
    """
    examples = load_mrc(data_path)
    texts = [example["text"] for example in examples]
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return texts, embeddings

if __name__ == "__main__":
    texts, embeddings = generate_embeddings()
    
    if texts and embeddings.any():
        print(f"\nЭмбеддинги успешно сгенерированы для {len(texts)} текстов.")
        print(f"Размерность одного эмбеддинга: {embeddings[0].shape}")
    else:
        print("Не удалось сгенерировать эмбеддинги. Проверьте входные данные и модель.") 