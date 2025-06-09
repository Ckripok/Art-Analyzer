import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import font_manager
import random
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
from transformers import BlipProcessor, BlipForConditionalGeneration
from scipy import stats
from skimage.measure import shannon_entropy
from numpy import mean, var, corrcoef
from skimage.feature import hog
from skimage import color
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from huggingface_hub import hf_hub_download

os.makedirs("./cache", exist_ok=True)
def get_style_examples(style_name, num=3):
    print("✅ get_style_examples стартует...")
    base_path = os.path.join(BASE_DIR, "..", "dataset_styles", "train", style_name)
    if not os.path.exists(base_path):
        return []
    files = [f for f in os.listdir(base_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(files, min(num, len(files)))
    return [f"/static_examples/{style_name}/{s}" for s in selected]

def get_genre_examples(genre_name, num=3):
    print("✅ get_genre_examples стартует...")
    base_path = os.path.join(BASE_DIR, "..", "dataset_genre", "train", genre_name)
    if not os.path.exists(base_path):
        return []
    files = [f for f in os.listdir(base_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(files, min(num, len(files)))
    return [f"/static_examples_genre/{genre_name}/{s}" for s in selected]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(BASE_DIR, "..", "static", "Font", "Tablon-Regular.ttf")

font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Tablon'

matplotlib.use('Agg')
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENRE_PATH = hf_hub_download(repo_id="DatsuNTOYOTA/ARTS", filename="model_genre.pth", cache_dir="./cache")
STYLE_PATH = hf_hub_download(repo_id="DatsuNTOYOTA/ARTS", filename="model_styles.pth", cache_dir="./cache")

try:
    genre_classes = sorted(os.listdir(os.path.join(BASE_DIR, "..", "dataset_genre", "train")))
except FileNotFoundError:
    genre_classes = []

try:
    style_classes = sorted(os.listdir(os.path.join(BASE_DIR, "..", "dataset_styles", "train")))
except FileNotFoundError:
    style_classes = []

blip_processor = None
blip_model = None

def load_blip_models():
    print("✅ load_blip_models стартует...")
    global blip_processor, blip_model
    if blip_processor is None or blip_model is None:
        print("📦 Загружаем BLIP модель...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_safetensors=True
        )
        print("✅ BLIP модель загружена.")


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_model(path, num_classes):
    print("✅ load_model стартует...")
    if num_classes == 0:
        return None  # модель не создаётся без классов

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model_genre = None
model_style = None

def ensure_models_loaded():
    global model_genre, model_style
    if model_genre is None:
        print("📦 Загружаем модель жанров...")
        if len(genre_classes) > 0:
            model_genre = load_model(GENRE_PATH, len(genre_classes))
    if model_style is None:
        print("🎨 Загружаем модель стилей...")
        if len(style_classes) > 0:
            model_style = load_model(STYLE_PATH, len(style_classes))


# Палитра
def extract_palette(image: Image.Image, n_colors=28):
    print("✅ extract_palette стартует...")
    image = image.resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)

    # Преобразование RGB в HSV и сортировка по Hue
    def rgb_to_hsv(rgb):
        print("✅ rgb_to_hsv стартует...")
        return colorsys.rgb_to_hsv(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)

    sorted_centers = sorted(centers, key=lambda c: rgb_to_hsv(c)[0])  # сортировка по оттенку (Hue)

    return ['#%02x%02x%02x' % tuple(c) for c in sorted_centers]

# CAM
#def generate_cam(image: Image.Image, output_path: str, model, class_idx):
    print("✅ generate_cam стартует...")
    cam_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = cam_transform(image).unsqueeze(0).to(DEVICE)
    features = []

    def hook_fn(module, input, output):
        print("✅ hook_fn стартует...")
        features.append(output.detach())

    hook = model.layer4[-1].register_forward_hook(hook_fn)
    _ = model(input_tensor)
    hook.remove()

    feature_maps = features[0].squeeze(0)
    weights = model.fc.weight[class_idx]  # более явно и безопасно
    cam = torch.zeros(feature_maps.shape[1:]).to(DEVICE)
    for i, w in enumerate(weights):
        cam += w * feature_maps[i]

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-5)
    cam = cv2.resize(cam, (image.width, image.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(image.convert("RGB"))[:, :, ::-1]
    result = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, result)

def plot_cumulative_and_pdf(data, output_cdf_path, output_pdf_path, color):
    print("✅ plot_cumulative_and_pdf стартует...")
    # Кумулятивная гистограмма
    plt.figure(figsize=(6, 3), dpi=100)
    hist, bins = np.histogram(data, bins=256, density=True)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    plt.plot(bins[1:], cdf, color=color)
    plt.title("Кумулятивная гистограмма", fontdict={'family': 'Tablon', 'color': '#00ff88', 'fontsize': 12})
    plt.gca().set_facecolor('none')
    plt.xticks(color='#aaa', fontname='Tablon')
    plt.yticks(color='#aaa', fontname='Tablon')
    plt.tight_layout()
    plt.savefig(output_cdf_path, transparent=True)
    plt.close()

    # Плотность распределения
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x = np.linspace(0, 1, 256)
    y = kde(x)
    plt.figure(figsize=(6, 3), dpi=100)
    plt.plot(x, y, color=color)
    plt.title("Плотность распределения (PDF)", fontdict={'family': 'Tablon', 'color': '#00ff88', 'fontsize': 12})
    plt.gca().set_facecolor('none')
    plt.xticks(color='#aaa', fontname='Tablon')
    plt.yticks(color='#aaa', fontname='Tablon')
    plt.tight_layout()
    plt.savefig(output_pdf_path, transparent=True)
    plt.close()


# Главная функция предсказания
def predict_image_top3(image: Image.Image, save_cam_path=None):
    print("✅ predict_image_top3 стартует...")
    ensure_models_loaded()
    saturation, brightness, contrast = calc_saturation_contrast(image)
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    saturation, brightness, contrast = calc_saturation_contrast(image)

    with torch.no_grad():
        genre_output = model_genre(input_tensor)
        style_output = model_style(input_tensor)

        genre_probs = F.softmax(genre_output, dim=1).cpu().numpy()[0]
        style_probs = F.softmax(style_output, dim=1).cpu().numpy()[0]

    if model_genre is None or model_style is None:
        return {
            "error": "Модели не загружены. Проверьте наличие dataset_genre/train и dataset_styles/train."
        }

    genre_top3 = sorted(zip(genre_classes, genre_probs), key=lambda x: x[1], reverse=True)
    style_top3 = sorted(zip(style_classes, style_probs), key=lambda x: x[1], reverse=True)

    #caption = generate_caption(image)
    caption = "Описание временно отключено"
    palette = extract_palette(image)

    avg_saturation = float(np.mean(saturation))
    avg_brightness = float(np.mean(brightness))
    width, height = image.size




    if save_cam_path:
        class_idx = np.argmax(genre_probs)
        generate_cam(image, save_cam_path, model_genre, class_idx)

    saturation, brightness, contrast = calc_saturation_contrast(image)

    img_np = np.array(image.convert("RGB")) / 255.0

    # === RGB статистика ===
    r_channel = img_np[..., 0].flatten()
    g_channel = img_np[..., 1].flatten()
    b_channel = img_np[..., 2].flatten()

    rgb_stats = {
        "mean": {
            "R": float(mean(r_channel)),
            "G": float(mean(g_channel)),
            "B": float(mean(b_channel))
        },
        "variance": {
            "R": float(var(r_channel)),
            "G": float(var(g_channel)),
            "B": float(var(b_channel))
        },
        "correlation": {
            "RG": float(corrcoef(r_channel, g_channel)[0, 1]),
            "RB": float(corrcoef(r_channel, b_channel)[0, 1]),
            "GB": float(corrcoef(g_channel, b_channel)[0, 1])
        }
    }

    # === HSV статистика ===
    hsv_np = matplotlib.colors.rgb_to_hsv(img_np)

    h_channel = hsv_np[..., 0].flatten()
    s_channel = hsv_np[..., 1].flatten()
    v_channel = hsv_np[..., 2].flatten()

    hsv_stats = {
        "mean": {
            "H": float(mean(h_channel)),
            "S": float(mean(s_channel)),
            "V": float(mean(v_channel))
        },
        "variance": {
            "H": float(var(h_channel)),
            "S": float(var(s_channel)),
            "V": float(var(v_channel))
        },
        "correlation": {
            "HS": float(corrcoef(h_channel, s_channel)[0, 1]),
            "HV": float(corrcoef(h_channel, v_channel)[0, 1]),
            "SV": float(corrcoef(s_channel, v_channel)[0, 1])
        }
    }

    # Сохраняем графики
    sat_path = "static/saturation_hist.png"
    bright_path = "static/brightness_hist.png"
    #plot_histogram(saturation, "Saturation Histogram", "#00ccef", sat_path)
    #plot_histogram(brightness, "Brightness Histogram", "#00f395", bright_path)
    cdf_path = "static/brightness_cdf.png"
    pdf_path = "static/brightness_pdf.png"
    #plot_cumulative_and_pdf(brightness, cdf_path, pdf_path, "#00ff88")

    bright_flat = brightness.flatten()
    brightness_stats = {
        "mean": float(np.mean(bright_flat)),
        "median": float(np.median(bright_flat)),
        "min": float(np.min(bright_flat)),
        "max": float(np.max(bright_flat)),
        "std": float(np.std(bright_flat)),
        "mode": float(stats.mode(bright_flat, keepdims=False).mode),
        "energy": float(np.sum(np.square(bright_flat))),
        "entropy": float(shannon_entropy(bright_flat))
    }

    gray = np.array(image.convert("L")) / 255.0
    fd, hog_image = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, channel_axis=None)

    hog_path = "static/hog_image.png"
    plt.figure(figsize=(6, 3), dpi=100)
    plt.imshow(hog_image, cmap="inferno")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(hog_path, transparent=True)
    plt.close()

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_path = "static/lbp_image.png"
    plt.figure(figsize=(6, 3), dpi=100)
    plt.imshow(lbp, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(lbp_path, transparent=True)
    plt.close()

    # Преобразование в 8-бит и расчет GLCM
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Извлечение признаков
    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_energy = graycoprops(glcm, 'energy')[0, 0]
    glcm_correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Энтропия GLCM
    glcm_entropy = -np.sum(glcm * np.log2(glcm + 1e-12))

    #orb = cv2.ORB_create(nfeatures=200)
    keypoints, descriptors = orb.detectAndCompute((gray * 255).astype(np.uint8), None)

    orb_img = cv2.drawKeypoints((gray * 255).astype(np.uint8), keypoints, None, color=(0, 255, 0))
    orb_path = "static/orb_image.png"
    cv2.imwrite(orb_path, orb_img)

    return {
        "genre_top3": [{"label": s[0],"confidence": round(float(s[1]) * 100, 2),"examples": get_genre_examples(s[0])}for s in genre_top3],
        "style_top3": [{"label": s[0],"confidence": round(float(s[1]) * 100, 2),"examples": get_style_examples(s[0])}for s in style_top3],
        "caption": caption,
        "palette": palette,
        "cam_path": save_cam_path if save_cam_path else None,
        "saturation_hist": os.path.basename(sat_path),
        "brightness_hist": os.path.basename(bright_path),
        "metadata": {
            "width": width,
            "height": height,
            "avg_saturation": round(avg_saturation, 3),
            "avg_brightness": round(avg_brightness, 3),
            "contrast": round(contrast, 3),
            "brightness_stats": brightness_stats,
            "color_statistics": {
                "rgb": rgb_stats,
                "hsv": hsv_stats
            },
            "cdf_path": os.path.basename(cdf_path),
            "pdf_path": os.path.basename(pdf_path),
            "hog_path": os.path.basename(hog_path),
            "texture_features": {
                "contrast": float(glcm_contrast),
                "homogeneity": float(glcm_homogeneity),
                "energy": float(glcm_energy),
                "entropy": float(glcm_entropy),
                "correlation": float(glcm_correlation)
            },
            "lbp_path": os.path.basename(lbp_path),
            "orb_path": os.path.basename(orb_path),


        }
    }


# Пример массового анализа
def predict_all(folder="manual_tests"):
    print("✅ predict_all стартует...")
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, file)
            image = Image.open(path).convert("RGB")
            cam_output = f"cam_{os.path.splitext(file)[0]}.jpg"
            result = predict_image_top3(image, cam_output)
            print(f"{file} — Genres: {result['genre_top3']} | Styles: {result['style_top3']}")


def calc_saturation_contrast(image: Image.Image):
    print("✅ calc_saturation_contrast стартует...")
    img_np = np.array(image.convert("RGB")) / 255.0
    hsv = matplotlib.colors.rgb_to_hsv(img_np)

    saturation = hsv[..., 1]
    value = hsv[..., 2]
    contrast = np.std(value)

    return saturation.flatten(), value.flatten(), contrast

def plot_histogram(data, title, color, output_path):
    print("✅ plot_histogram стартует...")
    plt.figure(figsize=(6, 3), dpi=100)
    plt.hist(data, bins=30, color=color, edgecolor='none', alpha=0.8)
    plt.title(title, fontdict={'family': 'Tablon', 'color': '#00ff88', 'fontsize': 12})
    plt.gca().set_facecolor('none')
    plt.grid(False)
    plt.xticks(color='#aaa', fontname='Tablon')
    plt.yticks(color='#aaa', fontname='Tablon')
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()
def generate_caption(image: Image.Image) -> str:
    print("✅ generate_caption стартует...")
    load_blip_models()
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(
        **inputs,
        max_length=100,
        num_beams=5,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True
    )
    return blip_processor.decode(out[0], skip_special_tokens=True)
