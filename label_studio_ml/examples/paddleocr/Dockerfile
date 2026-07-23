FROM python:3.10-slim-bullseye

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

# Installing libGL
RUN apt-get update && apt-get install -y \
    libgl1 libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6


RUN pip install --upgrade pip \
 && pip install -r requirements.txt
 

#COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

WORKDIR /app

COPY * /app/

EXPOSE 9090

CMD ["/usr/local/bin/supervisord", \
     "-c", \
     "/etc/supervisor/conf.d/supervisord.conf"]
