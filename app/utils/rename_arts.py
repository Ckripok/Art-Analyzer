import os
def rename_images_sequentially(folder_path, extension=".jpg"):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()
    count = 1
    for filename in files:
        old_path = os.path.join(folder_path, filename)
        new_name = f"{count}{extension}"
        new_path = os.path.join(folder_path, new_name)
        if old_path == new_path:
            count += 1
            continue
        try:
            os.rename(old_path, new_path)
            print(f"{filename} → {new_name}")
            count += 1
        except Exception as e:
            print(f"Ошибка при переименовании {filename}: {e}")

    print(f"\nПереименовано файлов: {count - 1}")
rename_images_sequentially("/styles(3)/Abstract", extension=".jpg")
