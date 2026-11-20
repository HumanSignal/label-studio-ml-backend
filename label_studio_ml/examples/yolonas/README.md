# YOLONAS ML Backend for Label Studio  

## Intro  
Use Deci AI [YOLONAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) model with Label Studio.  

## Setup
### 0. Important things to note  
 - This ML backend is designed to work in **docker**. You can run in on host but this manual does not cover that.  
 - There is no easy way to run ML backend with GPU support - you get an error `RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method`
 - Single image inference on CPU takes about a second.
 - Base docker image has a `jupyter lab` command. So you can comment a `command` in `docker-compose.yml`, uncomment `8888:8888` for port mapping and use it as jupyter lab with password `change-me` over https.
 - Main tested scenario has been for s3 cloud storage with custom endpoint url. Other storage options are not guaranteed to work. 
### 1. Clone this repo
### 2. Get model weights  
### 3. Adjust variables  
Adjust these variables in `.env` file.
```
CHECKPOINT_FILE="/home/testuser/app/model.pth"
PORT=9090
YOLO_LABELS=/home/testuser/app/labels.txt
IOU_THRESHOLD=0.25
SCORE_THRESHOLD=0.4
IMG_SIZE=1280
DEVICE=cpu
ENDPOINT_URL=<specify minio address http://myminio:9000>
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=<specify minio password>
LABEL_STUDIO_HOSTNAME=<specify label studio address with port like http://mylabelstudio.com:8080 >
YOLO_MODEL_TYPE=yolo_nas_m
```  

`YOLO_LABELS=/home/testuser/app/labels.txt` file with labels - each label on new line.  
Labels should be the same in labeling interface and in this file. 
If yolo labels differ you need to provide `LABELS_FILE` variable with mapping from Label studio label to yolo label like  `{"airplane": "Boeing"}`

### 4. Build docker image  
Run `docker compose build` to build an image.  
Base image `bodbe/yolonas` is built with [Dockerfile.full](Dockerfile.full).  

### 5. Run ML Backend  
Run `docker compose up -d` 

### 6. How to run on GPU  
 - Update DEVICE variable in `.env` file to `cuda:0`
 - Uncomment `deploy` section in `docker-compose.yml`
 - Change `command` section in `docker-compose.yml` to `bash -c "python _wsgi.py"
