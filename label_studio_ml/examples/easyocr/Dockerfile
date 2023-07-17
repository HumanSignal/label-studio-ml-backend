FROM python:3.8-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx

RUN pip install -r requirements.txt

RUN pip install -U opencv-python==4.5.5.64

COPY . ./

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
