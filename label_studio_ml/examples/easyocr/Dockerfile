FROM python:3.8-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app
COPY requirements.txt .

RUN apt-get update
RUN apt-get install libglib2.0-0

RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -r requirements.txt

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
