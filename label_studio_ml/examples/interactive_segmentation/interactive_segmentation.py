import os
import logging

import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path, get_image_size, get_single_tag_keys
from label_studio.core.utils.io import json_load, get_data_dir
from label_studio.core.settings.base import DATA_UNDEFINED_NAME

import torch
import cv2
import numpy as np

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils import exp
from isegm.inference import utils

logger = logging.getLogger(__name__)

class IteractiveSegmentation(LabelStudioMLBase):
    """Interactive segmentation detector """
    def __init__(self,
                 config_file,
                 checkpoint_file,
                 image_dir=None,
                 score_threshold=0.5,
                 device='cpu',
                 **kwargs):
        """
        Load segmentation model from config and checkpoint into memory.

        :param config_file: Absolute path to config file
        :param checkpoint_file: Absolute path to checkpoint file
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(IteractiveSegmentation, self).__init__(**kwargs)
        # config file and checkpoint for NN
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        print('Load new model from: ', config_file, checkpoint_file)
        # loading model from checkpoint
        self.device = torch.device(device)
        cfg = exp.load_config_file(config_file, return_edict=True)
        torch.backends.cudnn.deterministic = True

        # create model with arguments
        checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, checkpoint_file)
        self.model = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True)

        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        # load image
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        _result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        # loading context
        context = kwargs.get('context')
        x = context.get('x')
        y = context.get('y')
        is_positive = bool(context.get('is_positive'))
        # we need init mast if it's a second click
        _init_mask = context.get('mask')
        # add clicks for Clicker object
        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        _clicker = clicker.Clicker()
        _clicker.add_click(click)
        # get predictions
        pred_params = {'brs_mode': 'NoBRS',
                       'prob_thresh': 0.5,
                       'zoom_in_params': {'skip_clicks': -1, 'target_size': (400, 400), 'expansion_ratio': 1.4},
                       'predictor_params': {'net_clicks_limit': None, 'max_size': 800},
                       'brs_opt_func_params': {'min_iou_diff': 0.001},
                       'lbfgs_params': {'maxfun': 20}}
        predictor = get_predictor(self.model, device=self.device, **pred_params)
        predictor.set_input_image(image)
        pred = predictor.get_prediction(_clicker, prev_mask=_init_mask)

        # convert result mask to mask
        result_mask = _result_mask.copy()
        result_mask[pred > 0.5] = 255
        result_mask = result_mask.astype(np.uint8)

        return [{
            'result': result_mask
        }]