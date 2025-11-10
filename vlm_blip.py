import os
import csv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# CSV original
input_csv = "/Users/carlotafernandez/Desktop/UIE/4º/1º cuatri/Sistemas Interactivos Inteligentes/unidad V/práctica 3/dataset_UrbanScenes.csv"
output_csv = "generated_captions.csv"

# Carpeta donde están las imágenes
dataset_folder = "dataset"

# Cargar modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Función para generar caption
def generar_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Función para buscar la imagen en las subcarpetas
def encontrar_imagen(image_name, base_folder=dataset_folder):
    for root, dirs, files in os.walk(base_folder):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

# Leer CSV de entrada y generar captions
with open(input_csv, mode='r', newline='', encoding='utf-8') as f_in, \
     open(output_csv, mode='w', newline='', encoding='utf-8') as f_out:

    reader = csv.DictReader(f_in)
    fieldnames = ["imagen", "categoria", "caption_original", "caption_generada"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        imagen_path = encontrar_imagen(row["image"])
        categoria = row.get("category", "")
        caption_original = row.get("description", "")

        if imagen_path is None:
            print(f"Imagen no encontrada: {row['image']}")
            continue

        caption_generada = generar_caption(imagen_path)
        writer.writerow({
            "imagen": imagen_path,
            "categoria": categoria,
            "caption_original": caption_original,
            "caption_generada": caption_generada
        })
        print(f"Procesada: {imagen_path} -> {caption_generada}")

print("Generación de captions completada. CSV guardado en", output_csv)
