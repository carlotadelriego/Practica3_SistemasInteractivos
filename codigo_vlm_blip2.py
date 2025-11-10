import os
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# --------- DISPOSITIVO ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --------- MODELO Y PROCESSOR ---------
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
print("Cargando modelo y procesador:", MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME).to(device)

# --------- RUTAS ---------
CSV_PATH = "dataset_UrbanScenes.csv"
IMAGES_DIR = "./dataset"
OUT_CSV = "results_blip2.csv"

df = pd.read_csv(CSV_PATH)
generated = []

# --------- FUNCION PARA ENCONTRAR IMAGEN ---------
def find_image(image_name, base_dir):
    for root, dirs, files in os.walk(base_dir):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

# --------- GENERAR CAPTIONS ---------
for i, row in df.iterrows():
    image_name = row["image"]
    img_path = find_image(image_name, IMAGES_DIR)

    if img_path is None:
        print(f"NO ENCONTRADO: {image_name}")
        generated.append("")
        continue

    image = Image.open(img_path).convert("RGB")
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0]*ratio), int(image.size[1]*ratio))
        image = image.resize(new_size)

    prompt = "Describe the scene in detail."
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80)

    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(f"{image_name} -> {caption}")
    generated.append(caption)

# --------- GUARDAR RESULTADOS ---------
df["generated_caption"] = generated
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"Resultados guardados en: {OUT_CSV}")
