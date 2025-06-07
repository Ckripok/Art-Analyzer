from huggingface_hub import HfApi, HfFolder, upload_file

# Замените на своё имя и репозиторий
repo_id = "DatsuNTOYOTA/ARTS"

# Загружаем обе модели
upload_file(
    path_or_fileobj="models/model_styles.pth",
    path_in_repo="model_styles.pth",
    repo_id=repo_id,
    repo_type="model"
)

upload_file(
    path_or_fileobj="models/model_genre.pth",
    path_in_repo="model_genre.pth",
    repo_id=repo_id,
    repo_type="model"
)
