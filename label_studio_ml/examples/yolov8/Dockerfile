FROM python:3.11-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget && \
    apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev gcc

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt
RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
