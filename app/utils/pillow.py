# Сжатие изображений
import cv2
import os
input_root = r'...'
output_root = r'...'
MAX_WIDTH = 512
MAX_HEIGHT = 512
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                img = cv2.imread(input_path)
                if img is None:
                    print(f'Ошибка чтения: {input_path}')
                    continue
                h, w = img.shape[:2]
                scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)
                new_size = (int(w * scale), int(h * scale))
                resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
            except Exception as e:
                print(f'Ошибка обработки {input_path}: {e}')
print("Сжатие завершено!")