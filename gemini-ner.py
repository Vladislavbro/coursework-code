import google.generativeai as genai
from datasets import load_dataset
import json
import time
import re
from evaluate import load as load_metric
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from prompts import prompt


API_KEY = "AIzaSyDqwBO7fYRUtmWktEXnXTzn-RX67zO2Pi4"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemma-3n-e4b-it')

response = model.generate_content(prompt)

with open("data/response.txt", "w", encoding="utf-8") as f:
    f.write(response.text)