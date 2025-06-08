from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
import os 
import requests

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

HuggingFace_URL = "https://huggingface.co/Derein/identifikasi-jenis-ikan/resolve/main/model.keras"
model_path = "model.keras"
# Download model jika belum ada dari HuggingFace
if not os.path.exists(model_path):
    print("Model tidak ditemukan, mengunduh dari HuggingFace...")
    response = requests.get(HuggingFace_URL)
    with open(model_path, 'wb') as f:
        f.write(response.content)
else:
    print("Model ditemukan, tidak perlu mengunduh.")
    
# Load Model
model = tf.keras.models.load_model(model_path)  # type: ignore

def preprocess_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).resize((224,224))
    img_array = np.array(image)  # Convert PIL Image to NumPy array
    # img_array = img_array / 255.0 # Normalisasi Gambar (uncomment if needed)
    return np.expand_dims(img_array, axis=0) # Dimensi batch

@app.get("/")
async def root():
    return {"message":"API telah siap digunakan!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        selected_classes = ['Catfish','Tilapia','Gourami','Snakehead','Pangasius','Silver Carp','Big Head Carp','Grass Carp','Indian Carp','Silver Barb','Perch','Bangus', 'Freshwater Eel','Climbing Perch']
        content = await file.read()
        if not content:
            raise ValueError("Gambar kosong atau tidak berhasil dibaca!")
        MAX_FILE_SIZE = 5 * 1024 *1024 # Maksimal ukuran file 5 MB
        if len(content) > MAX_FILE_SIZE:
            raise ValueError("Ukuran file terlalu besar! Maksimal 5 MB.")
        input_tensor = preprocess_image(content) # Pra-proses gambar
        prediction = model.predict(input_tensor) # Prediksi gambar menggunakan model
        prediction = prediction.tolist() # Mengubah array menjadi list
        percentages = [round(p * 100, 2) for p in prediction[0]] # Menghitung nilai float menjadi persentase
        predict_percentage = dict(zip(selected_classes, percentages)) # Menggabungkan kelas dengan persentase
        top_5 = {k:v for k, v in sorted(predict_percentage.items(), key=lambda item:item[1], reverse=True)[:5]}
        label = selected_classes[np.argmax(prediction)] # Mendapatkan label kelas dengan nilai tertinggi
        return JSONResponse(content={"result": label, "predict":prediction, "predict_percentage":predict_percentage, "top_5":top_5}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
