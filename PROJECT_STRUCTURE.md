# Структура проекта

```
корневая директория/
│
├── config.py                  # Конфигурация экспериментов и путей
├── data_utils.py              # Функции по работе с JSON
├── prompt_generator.py        # Генерация prompt'ов для LLM и сохранение ответов
├── evaluation.py              # Оценка качества разметки
├── main.py                    # Основной файл для запуска ОДНОГО эксперимента (параметры в config.py)
│
├── data/
│   └── wikiann_100.json       # тестовая выборка
│   └── wikiann_18.json        # обучающая выборка
│   └── response.json          # Ответы LLM
│   └── prompts.py             # Файл с разными текстами промптов
├── results/
│   └── results.csv            # Результаты экспериментов в CSV
│
└── notebooks/
    └── analysis.ipynb         # Анализ и визуализация результатов
```

## Описание
- Все модули в корне — для логики экспериментов, подготовки данных, генерации prompt'ов и оценки.
- `main.py` используется для запуска одного конкретного эксперимента, параметры которого настраиваются вручную в `config.py`.
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
  - shots ∈ {0, 5, 10, 15}  
  - model = gemma-3n-e4b-it
  - post-verification = off  

---

## 2. Prompt-Style  
**Цель:** оценить, влияет ли формат подсказки (JSON-schema vs Chain-of-Thought).  
- **Параметры:**  
  - shots = 5  
  - mode = knn  
  - style ∈ {json-schema, cot}  
  - model = gemma-3n-e4b-it  
  - post-verification = off  

---

## 3. Model-Variant  
**Цель:** сравнить разные LLM.  
- **Параметры:**  
  - shots = best option from experiment 1, style = best option from experiment 2  
  - model ∈ {gemma-3n-e4b-it и еще какие-то модели}  
  - post-verification = off  

---

## 4. Post-Verification  
**Цель:** измерить прирост precision от self-verification.  
- **Параметры:**  
  - shots = best option from experiment 1, style = best option from experiment 2, model = best option from experiment 3
  - post-verification ∈ {off, on}  

---


## Данные и Метрики в базе данных

- **Shots**  
  Количество примеров, которые использовались в промпте.

- **Model Name**  
  Используемая модель LLM

- **Style**  
  Стиль промпта (JSON-schema или Chain-of-Thought)

- **Post-Verification**  
  Пост-обработка и верификация разметки (yes/no)

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