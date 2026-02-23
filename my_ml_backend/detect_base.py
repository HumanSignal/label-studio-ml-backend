import numpy as np
from PIL import Image
# import json
from pdf_piece.utils import (DEFAULT_CFG,label_save,DEVICE)
from pdf_piece.utils.s3_load import img_from_s3
# from urllib.parse import urlparse
from pdf_piece.utils.formats import Formats
# from .title_detect import detect_title_block
from ultralytics import YOLO
# from..utils.log import setup_logger
# logger = setup_logger()


# config = DEFAULT_CFG
# format=config.format
# save_piece=config.save_piece
# DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

# save predictions with different format
''' label studio format 
[{
    "data": {
        "image": "/static/samples/sample.jpg" 
    },

    "predictions": [{
        "model_version": "one",
        "score": 0.5,
        "result": [ {},....]}
'''   
def dynamic_threshold(results, threshold_percent=75):
    # Create a dictionary to hold box_cls and corresponding confidences
    class_confidences = {}
    
    # Loop through results to populate the dictionary
    for result in results:
        for box in result.boxes:
            box_conf = box.conf.item()
            box_cls = int(box.cls.item())
            
            # Add confidences to the respective class
            if box_cls not in class_confidences:
                class_confidences[box_cls] = []
            class_confidences[box_cls].append(box_conf)
    
    # Calculate the threshold for each class
    thresholds = {
        cls: np.percentile(confs, 100 - threshold_percent)
        for cls, confs in class_confidences.items()
    }
    
    return thresholds

def base_detect(overrides=None,**kwargs):
    image = kwargs.get('image',None)
    initial_id = kwargs.get("id",0)
    model_path = kwargs.get('model_path')
    image_url= kwargs.get('image_url',None)
    detect_image_url = kwargs.get('detect_image_url',None)
    original_width = kwargs.get('original_width')
    original_height = kwargs.get('original_height')
    #output_path = kwargs.get('output_path')
    #model_version = kwargs.get('model_version')
    labels = kwargs.get('labels')
    class_ids = kwargs.get('class_ids')
    threshold = kwargs.get('threshold',0.1)
    device = kwargs.get('device','0')
    filter_thresh = kwargs.get('filter_thresh',DEFAULT_CFG.thresholds)
    scale = kwargs.get('scale', 1.0) 
    format = kwargs.get('format',None) # Example of a default value
    save_piece=kwargs.get('save_piece', True)
    if detect_image_url:
        image = Image.open(detect_image_url)
    else:
        if image==None:
            image = img_from_s3(image_url)
        
    # scale the image if nessary, the original scale is 1.0
    image_width, image_height=image.size
    if scale!=1.0:
        scaled_width = int(image_width * scale)
        scaled_height = int(image_height * scale)
        scaled_image = image.resize((scaled_width, scaled_height))
    else:
        scaled_image=image

    if save_piece:
        original_width, original_height = image_width,image_height

    # Creating list for predictions and variable for scores
    predictions = []
    format_predictions=[]
    model=YOLO(model_path)
    results = model.predict(scaled_image, conf=threshold, 
                            classes=class_ids, 
                            device=device, verbose=False,
                            imgsz=2016)
    image.close()
    # results = results.to("cpu")
    default_threshold = dynamic_threshold(results,threshold_percent=80) 
    for result in results:
        for i, box in enumerate(result.boxes):                    
            box_conf = box.conf.item()
            box_cls = int(box.cls.item())
            box_pos = box.xyxy.tolist()
            x1, y1, x2, y2 = box_pos[0]
            label=labels[box_cls]
            params = {
                "i": i+initial_id,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "box_cls": box_cls,
                "box_conf": box_conf,
                "label": label,
                "original_width": original_width,
                "original_height": original_height,
                "scale": scale,
                "format": format,
                "image_url":image_url
            }   
            ## filter result 
            if filter_thresh:
                if box_conf>=default_threshold[box_cls]:
                #if box_conf >= filter_thresh[str(box_cls)]:
                    r=label_save(**params)
                    predictions.append(r)
                    if params["format"]=='label_studio' and \
                    params["label"]!="drawings":
                        fr= Formats(**params).label_studio_pred()
                        format_predictions.append(fr)
                    elif params["format"]=="diffgram":
                        fr = Formats(**params).diffgram_format()
                        format_predictions.append(fr)

                else:
                    pass
            
            else:
                r=label_save(**params)
                predictions.append(r)
    
    return predictions, format_predictions

def medium_detect(**kwargs): 
        # labels should come from the config/env set
        # labels=self.classes
        # the cropped image size 
        # image = kwargs.get('image')
        initial_id = kwargs.get("id",0)
        model_path = kwargs.get('model_path')
        image_url = kwargs.get('image_url',None)
        x=kwargs.get('x')
        y=kwargs.get('y')
        detect_image_url = kwargs.get('detect_image_url',None)
        original_width = kwargs.get('original_width')
        original_height = kwargs.get('original_height')
        labels = kwargs.get('labels')
        class_ids = kwargs.get('class_ids')
        threshold = kwargs.get('threshold',0.1)
        device = kwargs.get('device','0')
        if DEVICE=='cpu':
            device = 'cpu'
        filter_thresh = kwargs.get('filter_thresh',DEFAULT_CFG.thresholds)
        scale = kwargs.get('scale', 1.0)
        format = kwargs.get('format', None)
        save_piece=kwargs.get('save_piece', True)
        if detect_image_url:
            image = Image.open(detect_image_url)
        else:
            image = img_from_s3(image_url)

        if save_piece:
            original_width,original_height = image.size

        # Creating list for predictions and variable for scores
        predictions = []
        format_predictions=[]
        prediction_p = []
        format_prediction_p=[]

        # Getting prediction using model on the scaled image,for detecting some objects
        model=YOLO(model_path)
        results = model.predict(image, conf=threshold, classes=class_ids, device=device,verbose=False,imgsz=640)
        # results = results.to("cpu")
        image.close()
        default_threshold = dynamic_threshold(results,threshold_percent=80)
        for result in results:
            for i, box in enumerate(result.boxes):                    
                box_conf = box.conf.item()
                box_cls = int(box.cls.item())
                box_pos = box.xyxy.tolist()
                x1, y1, x2, y2 = box_pos[0]
                label=labels[box_cls]
                if save_piece:
                    params_p = {
                    "i": i+initial_id,
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "box_cls": box_cls,
                    "box_conf": box_conf,
                    "label": label,
                    "original_width": original_width,
                    "original_height": original_height,
                    "scale": scale,
                    "format": format,
                    "image_url":detect_image_url
                }
                    r_p = label_save(**params_p)
                    prediction_p.append(r_p)
                    if filter_thresh:
                        if box_conf>=default_threshold[box_cls]:
                            if params_p["format"]=='label_studio':
                                fr_p= Formats(**params_p).label_studio_pred()
                                format_prediction_p.append(fr_p)

                            elif params_p["format"]=='diffgram':
                                fr_p= Formats(**params_p).diffgram_format()
                                format_predictions.append(fr_p)
                        else:
                            pass
                
                    else:
                        if params_p["format"]=='label_studio':
                            fr_p= Formats(**params_p).label_studio_pred()
                            format_prediction_p.append(fr_p)
                else:    
                    # relocate the position of bbx 
                    params = {
                        "i": i,
                        "x1": x1+x,
                        "x2": x2+x,
                        "y1": y1+y,
                        "y2": y2+y,
                        "box_cls": box_cls,
                        "box_conf": box_conf,
                        "label": label,
                        "original_width": original_width,
                        "original_height": original_height,
                        "scale": scale,
                        "format": format
                    }

                    if filter_thresh:
                        if box_conf >= filter_thresh[str(box_cls)]:
                            r=label_save(**params)
                            predictions.append(r)
                            if params["format"]=='label_studio':
                                # fm=params
                                fr= Formats(**params).label_studio_pred()
                                format_predictions.append(fr)
                            elif params["format"]=='diffgram':
                                fr= Formats(**params).diffgram_format()
                                format_predictions.append(fr)
                        else:
                            pass
                    
                    else:
                        r=label_save(**params)
                        predictions.append(r)  

        if save_piece:
            return prediction_p,format_prediction_p

        else:
            return predictions,format_predictions

