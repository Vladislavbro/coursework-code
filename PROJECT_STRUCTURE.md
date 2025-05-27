# Структура проекта

```
корневая директория/
│
├── config.py                  # Конфигурация экспериментов и путей
├── data_utils.py              # Функции по работе с JSON
├── embedding_generator.py     # Генерация эмбеддингов
├── example_selector.py        # Выбор примеров для prompt'ов
├── prompt_generator.py        # Генерация prompt'ов для LLM
├── post_verification.py       # Пост-обработка и верификация разметки
├── run_experiment.py          # Запуск экспериментов
├── evaluation.py              # Оценка качества разметки
│
├── data/
│   ├── wikiann_train.json     # MRC-формат обучающей выборки
│   └── wikiann_test.json      # MRC-тестовая выборка
│   └── prompts.py             # Файл с разными текстами промптов
├── results/
│   ├── experiments.db         # SQLite-база результатов
│   └── results.csv            # (альтернатива: результаты в CSV)
│
└── notebooks/
    └── analysis.ipynb         # Анализ и визуализация результатов
```

## Описание
- Все модули в корне — для логики экспериментов, подготовки данных, генерации prompt'ов и оценки.
- Каталог `data/` — для исходных и подготовленных датасетов.
- Каталог `results/` — для хранения результатов и логов экспериментов.
- Каталог `notebooks/` — для анализа и визуализации. 


---

# План экспериментов

Каждый эксперимент фиксирует метрики **accuracy**, **precision**, **recall**, **F1**.

---

## 1. Number-of-Shots  
**Цель:** понять, как меняется качество при росте числа демонстраций в prompt.  
- **Параметры:**  
  - mode = random  
  - shots ∈ {0, 1, 3, 5}  
  - model = gemma-3n-e4b-it
  - post-verification = off  

---

## 2. Example-Selection  
**Цель:** сравнить random vs kNN-retrieval (по эмбеддингам).  
- **Параметры:**  
  - shots = 3  
  - mode ∈ {random, knn}  
  - model = gemma-3n-e4b-it  
  - post-verification = off  

---

## 3. Prompt-Style  
**Цель:** оценить, влияет ли формат подсказки (JSON-schema vs Chain-of-Thought).  
- **Параметры:**  
  - shots = 3  
  - mode = knn  
  - style ∈ {json-schema, cot}  
  - model = gemma-3n-e4b-it  
  - post-verification = off  

---

## 4. Model-Variant  
**Цель:** сравнить разные LLM-движки.  
- **Параметры:**  
  - shots = 3, mode = knn, style = json-schema  
  - model ∈ {gemma-3n-e4b-it и еще какие-то модели}  
  - post-verification = off  

---

## 5. Post-Verification  
**Цель:** измерить прирост precision от self-verification.  
- **Параметры:**  
  - shots = 3, mode = knn, style = json-schema, model = gemma-3n-e4b-it  
  - post-verification ∈ {off, on}  

---


## Данные и Метрики в базе данных

- **Промпт:**  
  Промпты, которые использовались в эксперименте.

- **Model Name**  
  Используемая модель 

- **Accuracy:**  
  Отдельно accuracy (точность) и их динамика.

- **Precision:**  
  Отдельно precision (чистота) и их динамика.

- **Recall:**  
  Отдельно recall (полнота) и их динамика.

- **F1**  
  Основные метрики качества для каждого эксперимента.

- **Token Usage:**  
  Число входных + выходных токенов 

- **Variance F1:**  
  Статистика разброса F1 при повторном прогоне одного и того же prompt (опционально)

- **Self-verification Gain:**  
  Прирост precision/recall до и после верификации. (опционально)