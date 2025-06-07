# подготовка датасетов к обучению  20/80
import os
import shutil
import random
from tqdm import tqdm
SOURCE_DIR = '../../zip_genre' # путь(папка)/ что необходимо поделить 20/80
TARGET_DIR = '../../dataset_genre' # путь(папка)/ куда необходимо записать
SPLIT_RATIO = 0.8  # 80% train, 20% val
def clear_previous_dataset():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(os.path.join(TARGET_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'val'), exist_ok=True)
def prepare_dataset():
    clear_previous_dataset()
    print("Копирование изображений...")
    for class_name in tqdm(os.listdir(SOURCE_DIR), desc="Жанры"):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split_point = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_point]
        val_images = images[split_point:]
        for phase, imgs in zip(['train', 'val'], [train_images, val_images]):
            phase_dir = os.path.join(TARGET_DIR, phase, class_name)
            os.makedirs(phase_dir, exist_ok=True)
            for img in tqdm(imgs, desc=f"→ {phase}/{class_name}", leave=False):
                src = os.path.join(class_path, img)
                dst = os.path.join(phase_dir, img)
                shutil.copy2(src, dst)
    print("Dataset split complete.\n")
def count_images(dataset_path):
    total = 0
    per_class = {}
    for root, dirs, files in os.walk(dataset_path):
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            class_name = os.path.basename(root)
            count = len(images)
            per_class[class_name] = count
            total += count
    return total, per_class
if __name__ == "__main__":
    prepare_dataset()
    print("Подсчёт изображений в dataset_genre/train:")
    train_total, train_counts = count_images('../../dataset_genre/train')
    print(f"Всего: {train_total}")
    for cls, count in sorted(train_counts.items()):
        print(f"{cls}: {count}")
    print("\nПодсчёт изображений в dataset_genre/val:")
    val_total, val_counts = count_images('../../dataset_genre/val')
    print(f"Всего: {val_total}")
    for cls, count in sorted(val_counts.items()):
        print(f"{cls}: {count}")