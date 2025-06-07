import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.predict_all_combined import predict_image_top3
from PIL import Image
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static_examples", StaticFiles(directory="E:/pythonProject/dataset_styles/train"), name="static_examples")
app.mount("/static_examples_genre", StaticFiles(directory="E:/pythonProject/dataset_genre/train"), name="static_examples_genre")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("templates/main.html")



@app.get("/main.html")
async def main_page():
    return FileResponse("templates/main.html")

@app.get("/genres.html")
async def genres_page():
    return FileResponse("templates/genres.html")

@app.get("/styles.html")
async def styles_page():
    return FileResponse("templates/styles.html")

@app.get("/api/genres")
async def get_genres():
    base_path = "/pythonProject/dataset_genre/train"
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

@app.get("/api/styles")
async def get_styles():
    base_path = "/pythonProject/dataset_styles/train"
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
    return FileResponse("templates/analysis.html")

@app.get("/about.html")
async def about():
    return FileResponse("templates/about.html")

@app.get("/contacts.html")
async def contacts():
    return FileResponse("templates/contacts.html")

@app.post("/predict_all_combined")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Сохраняем CAM-карту в static/
    cam_filename = f"{uuid.uuid4().hex}.jpg"
    cam_path = os.path.join("static", cam_filename)

    results = predict_image_top3(image, save_cam_path=cam_path)

    return JSONResponse(content={
        "filename": file.filename,
        **results,
        "cam_path": cam_filename  # только имя файла, чтобы в JS подставить `/static/${cam_path}`
    })
