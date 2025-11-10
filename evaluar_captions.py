import csv
from collections import defaultdict
import evaluate

# CSV GENERADO
csv_file = "generated_captions.csv"

# CARGAR MÉTRICAS
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

# ESTRUCTURAS PARA ALMACENAR PREDICCIONES Y REFERENCIAS
predictions = []
references = []
categories = defaultdict(lambda: {"preds": [], "refs": []})
examples = {} 

# LEER CSV Y ALMACENAR DATOS
with open(csv_file, mode='r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pred = row["caption_generada"]
        ref = row["caption_original"]
        cat = row["categoria"]

        predictions.append(pred)
        references.append(ref)

        categories[cat]["preds"].append(pred)
        categories[cat]["refs"].append(ref)

        if cat not in examples:
            examples[cat] = {
                "imagen": row["imagen"],
                "caption_original": ref,
                "caption_generada": pred
            }

# MÉTRICAS GENERALES
bleu_score = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
rouge_score = rouge_metric.compute(predictions=predictions, references=references)
meteor_score = meteor_metric.compute(predictions=predictions, references=references)

print("MÉTRICAS GENERALES")
print("BLEU:", bleu_score)
print("ROUGE:", rouge_score)
print("METEOR:", meteor_score)
print("\n")

# Métricas por categoría y ejemplo cualitativo
print("MÉTRICAS POR CATEGORÍA")
for cat, data in categories.items():
    bleu_cat = bleu_metric.compute(predictions=data["preds"], references=[[r] for r in data["refs"]])
    rouge_cat = rouge_metric.compute(predictions=data["preds"], references=data["refs"])
    meteor_cat = meteor_metric.compute(predictions=data["preds"], references=data["refs"])
    print(f"Categoría: {cat}")
    print(f"  BLEU: {bleu_cat}")
    print(f"  ROUGE: {rouge_cat}")
    print(f"  METEOR: {meteor_cat}\n")
    
    ex = examples[cat]
    print("  Ejemplo cualitativo (para análisis visual):")
    print(f"    Imagen: {ex['imagen']}")
    print(f"      Caption original: {ex['caption_original']}")
    print(f"      Caption generada: {ex['caption_generada']}\n")