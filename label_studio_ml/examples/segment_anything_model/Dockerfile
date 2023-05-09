FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    VITH_CHECKPOINT=/app/sam_vit_h_4b8939.pth \
    ONNX_CHECKPOINT=/app/sam_onnx_quantized_example.onnx

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget \
    && pip install --no-cache-dir -r requirements.txt \
    && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY onnxconverter.py .
RUN python onnxconverter.py .

COPY _wsgi.py .
COPY segment_anything_model.py .
EXPOSE 9090
COPY * /app/



ENV ACCESS_TOKEN=0c5e516d37ed2bc1d11ff5fc59ebaf5e0f756386

RUN pip install opencv-python

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
