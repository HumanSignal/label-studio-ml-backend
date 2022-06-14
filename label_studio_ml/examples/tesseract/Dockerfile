FROM python:3.8-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt 

COPY . ./

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
