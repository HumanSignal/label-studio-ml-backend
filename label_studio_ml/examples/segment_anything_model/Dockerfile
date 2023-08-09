FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=True \
    PORT=9090

COPY requirements.txt .

RUN apt-get update && apt-get install -y wget git python3-opencv \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 9090

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
