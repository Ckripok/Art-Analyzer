import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.predict_all_combined import predict_image_top3
from PIL import Image
import io
import logging
from fastapi.logger import logger

app = FastAPI()
print("✅ FastAPI стартует...")

app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO)


static_styles_path = os.path.join(BASE_DIR, "dataset_styles", "train")
if os.path.exists(static_styles_path):
    app.mount("/static_examples", StaticFiles(directory=static_styles_path), name="static_examples")

static_genre_path = os.path.join(BASE_DIR, "dataset_genre", "train")
if os.path.exists(static_genre_path):
    app.mount("/static_examples_genre", StaticFiles(directory=static_genre_path), name="static_examples_genre")

@app.get("/")
async def home():
    print("✅ main_home стартует...")
    return FileResponse(os.path.join(BASE_DIR, "..", "templates", "main.html"))


@app.get("/main.html")
async def main_page():
    print("✅ main_main_page стартует...")
    return FileResponse(os.path.join(BASE_DIR, "..", "templates", "main.html"))


@app.get("/genres.html")
async def genres_page():
    print("✅ main_genres_page стартует...")
    return FileResponse(os.path.join(BASE_DIR, "..", "templates", "genres.html"))


@app.get("/styles.html")
async def styles_page():
    print("✅ main_styles_page стартует...")
    return FileResponse(os.path.join(BASE_DIR, "..", "templates", "styles.html"))


@app.get("/api/genres")
async def get_genres():
    print("✅ main_get_genres стартует...")
    base_path = os.path.join(BASE_DIR, "..", "dataset_genre", "train")

    genres = []
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            files = [
                f"/static_examples_genre/{folder}/{f}"
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ][:10]
            genres.append({
                "name": folder,
                "description": f"Описание жанра {folder}.",
                "examples": files
            })

    return JSONResponse(content=genres)


@app.get("/debug/fs")
def debug_fs():
    print("✅ main_debug_fs стартует...")
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        "BASE_DIR": base,
        "dataset_genre_path": os.path.exists(os.path.join(base, "..", "dataset_genre", "train")),
        "dataset_styles_path": os.path.exists(os.path.join(base, "..", "dataset_styles", "train")),
    }



@app.get("/api/styles")
async def get_styles():
    print("✅ main_get_styles стартует...")
    base_path = os.path.join(BASE_DIR, "..", "dataset_styles", "train")
    styles = []
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            files = [
                f"/static_examples/{folder}/{f}"
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ][:10]
            styles.append({
                "name": folder,
                "description": f"Описание стиля {folder}.",
                "examples": files
            })

    return JSONResponse(content=styles)



@app.get("/analysis.html")
async def analysis_page():
    print("✅ analysis_page стартует...")
    return FileResponse(os.path.join(BASE_DIR, "..", "templates", "analysis.html"))


@app.get("/about.html")
async def about():
    print("✅ about стартует...")
    return FileResponse(os.path.join(BASE_DIR, "..", "templates", "about.html"))


@app.get("/contacts.html")
async def contacts():
    print("✅ contacts стартует...")
    return FileResponse(os.path.join(BASE_DIR, "..", "templates", "contacts.html"))


from fastapi import Request  # добавь импорт

@app.post("/predict_all_combined")
async def predict(file: UploadFile = File(...), request: Request = None):
    print("✅ predict стартует...")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        cam_filename = f"{uuid.uuid4().hex}.jpg"
        cam_path = os.path.join(BASE_DIR, "..", "static", cam_filename)

        print("▶️  Начинаем анализ изображения...")
        results = predict_image_top3(image, save_cam_path=cam_path)
        print("✅ Результат получен:", results)

        if "error" in results:
            logger.error(f"❌ Ошибка в анализе: {results['error']}")
            return JSONResponse(status_code=500, content=results)

        return JSONResponse(content={
            "filename": file.filename,
            **results,
            "cam_path": cam_filename
        })


    except Exception as e:

        import traceback

        traceback_str = traceback.format_exc()

        logger.error(f"❌ Exception: {str(e)}\n{traceback_str}")

        print(f"❌ Exception: {str(e)}\n{traceback_str}")  # <--- ДОБАВЬ ЭТО

        return JSONResponse(status_code=500, content={"error": str(e)})



