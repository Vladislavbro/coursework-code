# %%
# !pip install evaluate seqeval

# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from evaluate import load as load_metric
import json

# %%
# 1. Загрузка датасета WikiANN
# dataset = load_dataset("wikiann", "ru") # Было для русского языка
dataset = load_dataset("wikiann", "en") # Теперь для английского языка

# Получаем список меток
label_list = dataset["train"].features["ner_tags"].feature.names
train_small = dataset["train"].shuffle(seed=42).select(range(2000))
val_small   = dataset["validation"].shuffle(seed=42).select(range(100))
train_sample_20 = train_small.shuffle(seed=42).select(range(20))


# %%
# tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased") # Был для русского языка
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") # Теперь для английского языка

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train = train_small.map(tokenize_and_align_labels, batched=True)
tokenized_val   = val_small.map(tokenize_and_align_labels, batched=True)

# %%
# 3. Загрузка модели и настройка Trainer
# model = AutoModelForTokenClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=len(label_list)) # Была для русского языка
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list)) # Теперь для английского языка
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

# %%
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels = [[label_list[l] for l in lab if l != -100] for lab in labels]
    true_preds  = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# %%
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,

    report_to=[],        # ← отключаем все логгеры (включая wandb)
    logging_steps=100,
    eval_steps=500,
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# %%
print(f"Train samples: {len(tokenized_train)}")
print(f"Val samples:   {len(tokenized_val)}")

# %%
# 4. Обучение и оценка
trainer.train()
metrics = trainer.evaluate()

# %%
metrics

# %%
# Конвертация в формат: текст + список сущностей с лейблом
def convert(example):
    tokens = example["tokens"]
    tags = example["ner_tags"]

    text = " ".join(tokens)
    entities = []

    current_entity = []
    current_label = None

    for token, tag_id in zip(tokens, tags):
        tag = label_list[tag_id]
        if tag.startswith("B-"):
            if current_entity:
                entities.append({
                    "text": " ".join(current_entity),
                    "label": current_label
                })
            current_entity = [token]
            current_label = tag[2:]
        elif tag.startswith("I-") and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append({
                    "text": " ".join(current_entity),
                    "label": current_label
                })
                current_entity = []
                current_label = None

    # Добавим последнюю сущность, если осталась
    if current_entity:
        entities.append({
            "text": " ".join(current_entity),
            "label": current_label
        })

    return {"text": text, "entities": entities}

# %%
# Применим ко всем примерам
converted = [convert(ex) for ex in val_small]

# Сохраняем в файл
# with open("wikiann_100.json", "w", encoding="utf-8") as f: # Было для русского
with open("wikiann_en_100.json", "w", encoding="utf-8") as f: # Теперь для английского
    json.dump(converted, f, indent=2, ensure_ascii=False)

# %%
# json для обучения Гемини
converted_20 = [convert(ex) for ex in train_sample_20]

# Сохраняем в файл
# with open("wikiann_20.json", "w", encoding="utf-8") as f: # Было для русского
with open("wikiann_en_20.json", "w", encoding="utf-8") as f: # Теперь для английского
    json.dump(converted_20, f, indent=2, ensure_ascii=False)

# %%
# with open("eval_metrics.json", "w") as f: # Было для русского
with open("eval_metrics_en.json", "w") as f: # Теперь для английского
    json.dump(metrics, f, indent=2)


