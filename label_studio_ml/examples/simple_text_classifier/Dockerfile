FROM python:3.8-slim

ENV PYTHONUNBUFFERED=True

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 9090

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
