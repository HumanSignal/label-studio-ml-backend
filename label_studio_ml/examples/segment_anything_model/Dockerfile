FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=True \
    PORT=9090

COPY requirements.txt .

RUN apt-get update && apt-get install -y wget git python3-opencv \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

# Make the start.sh script executable
RUN chmod +x /app/start.sh

EXPOSE 9090

# Set the start.sh script as the entrypoint
ENTRYPOINT ["/app/start.sh"]
