# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copiar scripts y dataset
COPY . /app

# Instalar pip y dependencias
RUN pip install --upgrade pip

# PyTorch CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Librerías necesarias
RUN pip install transformers evaluate pillow absl-py nltk rouge_score

# Descargar recursos de nltk para ROUGE
RUN python -m nltk.downloader punkt

# Comando por defecto: ejecutar evaluación
CMD ["python", "evaluar_captions.py"]
