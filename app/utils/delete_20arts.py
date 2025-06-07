import os
import random
base_dir = r"E:\pythonProject\dataset_styles\val"
keep_count = 20
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

for genre_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, genre_folder)

    if os.path.isdir(folder_path):
        all_images = [f for f in os.listdir(folder_path)
                      if os.path.splitext(f)[1].lower() in image_extensions]
        if len(all_images) > keep_count:
            images_to_keep = set(random.sample(all_images, keep_count))

            for image in all_images:
                if image not in images_to_keep:
                    image_path = os.path.join(folder_path, image)
                    try:
                        os.remove(image_path)
                        print(f"Удалено: {image_path}")
                    except Exception as e:
                        print(f"Ошибка при удалении {image_path}: {e}")
        else:
            print(f"Пропущено: {folder_path} (изображений меньше или ровно {keep_count})")
