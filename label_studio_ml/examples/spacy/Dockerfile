FROM python:3.8-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    SPACY_MODEL=en_core_web_sm

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download $SPACY_MODEL

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
