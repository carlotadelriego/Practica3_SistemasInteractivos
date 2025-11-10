# Dockerfile
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar scripts y dataset
COPY . /app

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers evaluate pillow

# Comando por defecto: generar captions y evaluar
CMD ["python", "evaluar_final.py"]
