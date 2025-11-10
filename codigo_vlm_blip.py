# codigo_vlm_blip.py
import os
import pandas as pd
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# CONFIGURACIÓN DE DISPOSITIVO
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# MODELO Y PROCESSOR
MODEL_NAME = "Salesforce/blip-image-captioning-base"

print("Cargando modelo y procesador:", MODEL_NAME)
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# RUTAS Y ARCHIVOS
CSV_PATH = "dataset_UrbanScenes.csv"
IMAGES_DIR = "./dataset"
OUT_CSV = "resultados_captions_blip.csv"

# LEER EL CSV
df = pd.read_csv(CSV_PATH)
generated = []

# GENERAR CAPTIONS PARA CADA IMAGEN
for i, row in df.iterrows():
    image_name = row["image"]
    img_path = os.path.join(IMAGES_DIR, image_name)

    if not os.path.exists(img_path):
        print(f"WARNING: {img_path} no existe — saltando")
        generated.append("")
        continue

    image = Image.open(img_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=64, num_beams=5)
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    print(f"{image_name} -> {caption}")
    generated.append(caption)

# AÑADIR CAPTIONS AL DATAFRAME Y GUARDAR
df["generar_caption"] = generated
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"Resultados guardados en: {OUT_CSV}")
