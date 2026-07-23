FROM python:3.11-slim

WORKDIR /app

RUN apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev gcc

# Install Base
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
