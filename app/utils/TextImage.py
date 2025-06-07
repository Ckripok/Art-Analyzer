#  просто проверки работы BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from tqdm import tqdm
import time

# Загрузка модели и процессора
print("Загрузка модели...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Загрузка изображения
image = Image.open("/dataset_styles/train/Academic/0j932has.jpg").convert('RGB')

# Эмуляция анализа с прогресс-баром
print("Анализ изображения:")
for _ in tqdm(range(5), desc="Обработка"):
    time.sleep(0.3)  # Имитируем прогресс (можно убрать, если не нужен эффект)

# Генерация описания
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("\nРезультат:", caption)
