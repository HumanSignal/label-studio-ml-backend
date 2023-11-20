FROM python:3.11-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget && \
    apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev gcc

ENV PYTHONUNBUFFERED=True \
    PORT=9090 

WORKDIR /tmp
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt
RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

RUN mkdir -p datasets/temp/images
RUN mkdir -p datasets/temp/labels

COPY * /app/

EXPOSE 9090

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]