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

## 2. Difficult-Examples  
**Цель:** оценить влияние добавления сложных примеров в промпт на качество разметки.  
- **Параметры:**  
  - shots = best option from experiment 1  
  - difficult_examples ∈ {without, with}  
  - model = gemma-3n-e4b-it  
  - post-verification = off  

---

## 3. Prompt-Style  
**Цель:** оценить, влияет ли формат подсказки (JSON-schema vs Chain-of-Thought).  
- **Параметры:**  
  - shots = best option from experiment 1  
  - difficult_examples = best option from experiment 2  
  - style ∈ {json-schema, cot}  
  - model = gemma-3n-e4b-it  
  - post-verification = off  

---

## 4. Model-Variant  
**Цель:** сравнить разные LLM.  
- **Параметры:**  
  - shots = best option from experiment 1  
  - difficult_examples = best option from experiment 2  
  - style = best option from experiment 3  
  - model ∈ {gemma-3n-e4b-it и еще какие-то модели}  
  - post-verification = off  

---

## 5. Post-Verification  
**Цель:** измерить прирост precision от self-verification.  
- **Параметры:**  
  - shots = best option from experiment 1  
  - difficult_examples = best option from experiment 2  
  - style = best option from experiment 3  
  - model = best option from experiment 4  
  - post-verification ∈ {off, on}  

---