FROM python:3.10

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

# Installing libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev

RUN apt-get install -y poppler-utils libpoppler-cpp-dev libglib2.0-0


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
