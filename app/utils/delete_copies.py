# Удаляем дубли ищображений через хэш

import os
import hashlib
from PIL import Image
def get_image_hash(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((128, 128))
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None
def remove_duplicate_images(folder_path):
    hashes = {}
    deleted = 0
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        image_hash = get_image_hash(filepath)
        if image_hash is None:
            continue
        if image_hash in hashes:
            os.remove(filepath)
            deleted += 1
            print(f"Удалено: {filepath}")
        else:
            hashes[image_hash] = filepath
    print(f"\nУдалено дубликатов: {deleted}")
remove_duplicate_images("...") #- путь к папке
