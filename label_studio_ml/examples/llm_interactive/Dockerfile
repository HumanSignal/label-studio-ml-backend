FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git tesseract-ocr

ENV PORT=9090

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
