# Makefile
IMAGE_NAME=vlm_practica3

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm -v "$(PWD)/dataset:/app/dataset" $(IMAGE_NAME)
