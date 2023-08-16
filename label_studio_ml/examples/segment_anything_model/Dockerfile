FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    MODEL_DIR=/data/models \
    VITH_CHECKPOINT=/app/models/sam_vit_h_4b8939.pth \
    MOBILESAM_CHECKPOINT=/app/models/mobile_sam.pt \
    ONNX_CHECKPOINT=/app/models/sam_onnx_quantized_example.onnx

RUN apt-get update -q \
    && apt-get install -qy --no-install-recommends wget git libopencv-dev python3-opencv \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and run the model download script
COPY download_models.sh .
RUN bash /app/download_models.sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Add ONNX model
RUN python3 onnxconverter.py

EXPOSE 9090

CMD ["/app/start.sh"]
