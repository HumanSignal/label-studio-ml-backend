# check if file `sam_vit_h_4b8939.pth` exists, otherwise download the model
[ -f sam_vit_h_4b8939.pth ] ||
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# check if file `mobile_sam.pt` exists, otherwise download the model
[ -f mobile_sam.pt ] ||
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# run ONNX conversion
python onnxconverter.py
