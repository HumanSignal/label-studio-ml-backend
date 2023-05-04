FROM python:3.8-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    VITH_CHECKPOINT=$(pwd)/sam_vit_h_4b8939.pth \
    ONNX_CHECKPOINT=$(pwd)/sam_onnx_quantized_example.onnx

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN curl -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY onnxconverter.py .
RUN python onnxconverter.py .

COPY _wsgi.py .
COPY segment_anything_model.py .
EXPOSE 9090

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
