FROM python:slim-bullseye

WORKDIR /tmp
COPY requirements.txt .

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

# Update the base OS and install Tesseract
RUN apt update -y \
 && apt upgrade -y
RUN apt install tesseract-ocr git -y
RUN apt install tesseract-ocr-chi-sim
RUN apt install tesseract-ocr-chi-tra
RUN apt install tesseract-ocr-deu

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
