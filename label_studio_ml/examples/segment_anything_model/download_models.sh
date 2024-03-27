#!/bin/bash

MODELS_DIR="models"
mkdir -p ${MODELS_DIR}

download_model() {
  FILE_PATH="${MODELS_DIR}/$1"
  URL="$2"

  if [ ! -f "${FILE_PATH}" ]; then
    wget -q "${URL}" -P ${MODELS_DIR}/
  fi
}

# Model files and their corresponding URLs
declare -A MODELS
MODELS["sam_vit_h_4b8939.pth"]="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODELS["mobile_sam.pt"]="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

for model in "${!MODELS[@]}"; do
  echo "Downloading ${model} to ${MODELS_DIR}/..."
  download_model "${model}" "${MODELS[${model}]}"
done
