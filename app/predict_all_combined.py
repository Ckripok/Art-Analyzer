from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import hf_hub_download

# === Параметры ===
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Классы ===
try:
    genre_classes = sorted(os.listdir(os.path.join(BASE_DIR, "..", "dataset_genre", "train")))
except FileNotFoundError:
    genre_classes = []

try:
    style_classes = sorted(os.listdir(os.path.join(BASE_DIR, "..", "dataset_styles", "train")))
except FileNotFoundError:
    style_classes = []

# === Модели ===
GENRE_PATH = hf_hub_download(repo_id="DatsuNTOYOTA/ARTS", filename="model_genre.pth", cache_dir="./cache")
STYLE_PATH = hf_hub_download(repo_id="DatsuNTOYOTA/ARTS", filename="model_styles.pth", cache_dir="./cache")

model_genre = None
model_style = None

def load_model(path, num_classes):
    print("✅ load_model стартует...")
    if num_classes == 0:
        return None
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def ensure_models_loaded():
    global model_genre, model_style
    if model_genre is None:
        print("📦 Загружаем модель жанров...")
        model_genre = load_model(GENRE_PATH, len(genre_classes))
    if model_style is None:
        print("🎨 Загружаем модель стилей...")
        model_style = load_model(STYLE_PATH, len(style_classes))

# === Преобразование ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Палитра ===
def extract_palette(image: Image.Image, n_colors=8):
    image = image.resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)
    from sklearn.cluster import KMeans
    import colorsys
    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    def rgb_to_hsv(rgb):
        return colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    sorted_centers = sorted(centers, key=lambda c: rgb_to_hsv(c)[0])
    return ['#%02x%02x%02x' % tuple(c) for c in sorted_centers]

# === Основная функция ===
def predict_image_top3(image: Image.Image):
    print("✅ predict_image_top3 стартует...")
    ensure_models_loaded()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        genre_output = model_genre(input_tensor)
        style_output = model_style(input_tensor)
        genre_probs = F.softmax(genre_output, dim=1).cpu().numpy()[0]
        style_probs = F.softmax(style_output, dim=1).cpu().numpy()[0]

    genre_top3 = sorted(zip(genre_classes, genre_probs), key=lambda x: x[1], reverse=True)[:3]
    style_top3 = sorted(zip(style_classes, style_probs), key=lambda x: x[1], reverse=True)[:3]
    palette = extract_palette(image)

    return {
        "genre_top3": [{"label": g[0], "confidence": round(float(g[1]) * 100, 2)} for g in genre_top3],
        "style_top3": [{"label": s[0], "confidence": round(float(s[1]) * 100, 2)} for s in style_top3],
        "palette": palette,
        "caption": "BLIP отключён",
        "metadata": {
            "width": image.width,
            "height": image.height
        }
    }
