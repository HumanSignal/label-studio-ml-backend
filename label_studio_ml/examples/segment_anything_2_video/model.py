import os
import pathlib
import tempfile
import logging
import json
from typing import List, Dict, Optional, Literal, cast
import sys

import cv2
import torch
import numpy as np
import requests
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.label_interface.objects import PredictionValue
# from PIL import Image
from collections import defaultdict
from label_studio_sdk.client import LabelStudio

# read the environment variables and set the paths just before importing the sam2 module
SEGMENT_ANYTHING_2_REPO_PATH = os.getenv('SEGMENT_ANYTHING_2_REPO_PATH', 'segment-anything-2')
sys.path.append(SEGMENT_ANYTHING_2_REPO_PATH)
from sam2.build_sam import build_sam2, build_sam2_video_predictor

logger = logging.getLogger(__name__)

DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'sam2_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2_hiera_large.pt')
MAX_FRAMES_TO_TRACK = int(os.getenv('MAX_FRAMES_TO_TRACK', 10))
PROMPT_TYPE = cast(Literal["box", "point"], os.getenv('PROMPT_TYPE', 'box'))
ANNOTATION_WORKAROUND = os.getenv('ANNOTATION_WORKAROUND', False)
DEBUG = os.getenv('DEBUG', False)

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam2_checkpoint = str(pathlib.Path(__file__).parent / SEGMENT_ANYTHING_2_REPO_PATH / "checkpoints" / MODEL_CHECKPOINT)
logger.debug(f'Model checkpoint: {sam2_checkpoint}')
logger.debug(f'Model config: {MODEL_CONFIG}')
predictor = build_sam2_video_predictor(MODEL_CONFIG, sam2_checkpoint)


# manage cache for inference state
# TODO: make it process-safe and implement cache invalidation
_predictor_state_key = ''
_inference_state = None

def get_inference_state(video_dir):
    global _predictor_state_key, _inference_state
    if _predictor_state_key != video_dir:
        _predictor_state_key = video_dir
        _inference_state = predictor.init_state(video_path=video_dir)
    return _inference_state

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def split_frames(self, video_path, temp_dir, start_frame=0, end_frame=100):
        logger.debug(f'Opening video file: {video_path}')
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        logger.debug(f'fps: {fps}, frame_count: {frame_count}')
        duration = frame_count / fps
        print(f'duration: {duration}')

        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        logger.debug(f'Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}')

        frame_count = 0
        while True:
            success, frame = video.read()

            if not success:
                logger.error(f'Failed to read frame {frame_count}')
                # manage this (frame 57 of acutal video test)
                # poi risovli il problema del label con diverse etichette
                break

            if frame_count < start_frame:
                frame_count += 1
                continue

            if frame_count >= end_frame:
                break

            frame_filename = os.path.join(temp_dir, f'{frame_count:05d}.jpg')

            if not os.path.exists(frame_filename):
                cv2.imwrite(frame_filename, frame)

            logger.debug(f'Frame {frame_count}: {frame_filename}')
            yield frame_filename, frame
            frame_count += 1

        video.release()

    def get_prompts(self, context) -> List[Dict]:
        logger.debug(f'Extracting keypoints from context: {context}')
        prompts = []
        for ctx in context['result']:
            # Process each video tracking object separately
            obj_id = ctx['id']
            for obj in ctx['value']['sequence']:
                x = obj['x'] / 100
                y = obj['y'] / 100
                box_width = obj['width'] / 100
                box_height = obj['height'] / 100
                frame_idx = obj['frame'] - 1

                if PROMPT_TYPE == 'point':
                    # SAM2 video works with keypoints - convert the rectangle to the set of keypoints within the rectangle
                    # bbox (x, y) is top-left corner
                    kps = [
                        # center of the bbox
                        [x + box_width / 2, y + box_height / 2],
                        # half of the bbox width to the left
                        [x + box_width / 4, y + box_height / 2],
                        # half of the bbox width to the right
                        [x + 3 * box_width / 4, y + box_height / 2],
                        # half of the bbox height to the top
                        [x + box_width / 2, y + box_height / 4],
                        # half of the bbox height to the bottom
                        [x + box_width / 2, y + 3 * box_height / 4]
                    ]
                elif PROMPT_TYPE == 'box':
                    # SAM2 video works with boxes - use the rectangle inf xyxy format
                    kps = [x, y, x + box_width, y + box_height]
                else:
                    raise ValueError(f'Invalid prompt type: {PROMPT_TYPE}')

                points = np.array(kps, dtype=np.float32)
                # labels are not used for box prompts
                labels = np.array([1] * len(kps), dtype=np.int32) if PROMPT_TYPE == 'point' else None
                prompts.append({
                    'points': points,
                    'labels': labels,
                    'frame_idx': frame_idx,
                    'obj_id': obj_id
                })

        return prompts

    def _get_fps(self, context):
        # get the fps from the context
        frames_count = context['result'][0]['value']['framesCount']
        duration = context['result'][0]['value']['duration']
        return frames_count, duration

    # def convert_mask_to_bbox(self, mask):
    #     # convert mask to bbox
    #     h, w = mask.shape[-2:]
    #     mask_int = mask.reshape(h, w, 1).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 0:
    #         return None
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     return {
    #         'x': x,
    #         'y': y,
    #         'width': w,
    #         'height': h
    #     }

    def convert_mask_to_bbox(self, mask):
        # squeeze
        mask = mask.squeeze()

        y_indices, x_indices = np.where(mask == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        # Find the min and max indices
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Get mask dimensions
        height, width = mask.shape

        # Calculate bounding box dimensions
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # Normalize and scale to percentage
        x_pct = (xmin / width) * 100
        y_pct = (ymin / height) * 100
        width_pct = (box_width / width) * 100
        height_pct = (box_height / height) * 100

        return {
            "x": round(x_pct, 2),
            "y": round(y_pct, 2),
            "width": round(width_pct, 2),
            "height": round(height_pct, 2)
        }


    def dump_image_with_mask(self, frame, mask, output_file, obj_id=None, random_color=False):
        from matplotlib import pyplot as plt
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # create an image file to display image overlayed with mask
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2BGR)
        mask_image = cv2.addWeighted(frame, 1.0, mask_image, 0.8, 0)
        logger.debug(f'Shapes: frame={frame.shape}, mask={mask.shape}, mask_image={mask_image.shape}')
        # save in file
        logger.debug(f'Saving image with mask to {output_file}')
        cv2.imwrite(output_file, mask_image)


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        Returns the predicted mask for a smart keypoint that has been placed.

        This function is responsible for processing video annotation tasks and predicting the mask of an object for a given video frame. It uses Label Studio context and draft data to determine the bounding boxes or keypoints that need to be predicted. The prediction is performed using a video tracking model, which processes multiple frames to create a coherent annotation for the target object across a sequence of video frames.

        For multi-object tracking, it is necessary to refer to the drafts instead of the context because the context contains only the data of the box that was most recently modified.

        The logic is as follows: each time the model is called, the prediction starts from the frame containing the last label of the object that appears the earliest in the video. By calling the model multiple times, the prediction is always performed moving forward.

        Steps involved in the process:
        1. Extract the relevant data from `tasks` and `context` to determine the prompts for the model.
        2. Cache the video locally and extract relevant frames using `split_frames`.
        3. Use the prompts to guide the model in identifying and tracking the object of interest.
        4. Generate a mask for each frame where the object is detected and track the object through subsequent frames.
        5. Propagate the detected objects through the video sequence to refine annotations and maintain consistency.
        6. Create or update the annotation in Label Studio to provide feedback to the user.

        Args:
            tasks (List[Dict]): List of tasks that need annotation.
            context (Optional[Dict]): Additional information about the current annotation context.
            kwargs: Optional additional arguments.

        Returns:
            ModelResponse: Response containing predicted annotations for the video frames.
        """
        from_name, to_name, value = self.get_first_tag_occurence('VideoRectangle', 'Video')
        # todo il problema è che label studio sbaglia la visualizzazione del primo frame: ho aperto una issue https://github.com/HumanSignal/label-studio/issues/6593 e anche una ocnversazione su slack
        try:
            drafts = tasks[0]['drafts'][0]
        except IndexError:
            logger.error('Drafts not found, using context')
            try:
                drafts = tasks[0]['annotations'][0]
            except IndexError:
                logger.error('Annotations not found, using context')
                drafts = context
        if not len(drafts):
            logger.info('Draft empty, using context')
            drafts = context
        task = tasks[0]
        task_id = task['id']
        # Get the video URL from the task
        video_url = task['data'][value]

        # cache the video locally
        video_path = get_local_path(video_url, task_id=task_id)
        logger.debug(f'Video path: {video_path}')

        # get prompts from context
        # prompts = self.get_prompts(context)
        prompts = self.get_prompts(drafts)

        context_ids = set([ctx['id'] for ctx in context['result']])
        all_obj_ids = set([p['id'] for p in drafts['result']] +
                          ([p['id'] for p in tasks[0]['annotations'][0]['result']] if len(tasks[0]['annotations']) else []))
        if not context_ids.issubset( all_obj_ids):
            # Returning here because the case where object ids in the context do not match the ids found in the annotations is not supported.
            # This remains an open issue but is not considered a substantial problem.
            raise NotImplementedError(f'Context id {context_ids} not found in drafts result: {all_obj_ids}'
                                      f'TODO merge context and drafts')

        # create a map from obj_id to integer
        obj_ids = {obj_id: i for i, obj_id in enumerate(all_obj_ids)}
        # find the last frame index
        # if there is only one object, use the last frame of the object: continue tracking from last tracked frame
        # if there are multiple objects, use the smallest frame index of all objects
        if len(all_obj_ids) == 1:
            first_frame_idx = min(p['frame_idx'] for p in prompts) if prompts else 0
            last_frame_idx = max(p['frame_idx'] for p in prompts) if prompts else 0
        else:
            first_frame_idx = min(p['frame_idx'] for p in prompts) if prompts else 0
            # the minimum of the maximum frame_idx of all objects grouped by id
            last_frame_idx = min(max(p['frame_idx'] for p in prompts if p['obj_id'] == obj_id) for obj_id in all_obj_ids)
        frames_count, duration = self._get_fps(context)
        fps = frames_count / duration

        logger.debug(
            f'Number of prompts: {len(prompts)}, '
            f'first frame index: {first_frame_idx}, '
            f'last frame index: {last_frame_idx}, '
            f'obj_ids: {obj_ids}')

        frames_to_track = min(MAX_FRAMES_TO_TRACK, frames_count - last_frame_idx)

        # Split the video into frames
        with tempfile.TemporaryDirectory() as temp_dir:

            # # use persisted dir for debug
            # temp_dir = '/tmp/frames'
            # os.makedirs(temp_dir, exist_ok=True)

            # get all frames
            frames = list(self.split_frames(
                video_path, temp_dir,
                start_frame=first_frame_idx,
                end_frame=last_frame_idx + frames_to_track
            ))
            height, width, _ = frames[0][1].shape
            logger.debug(f'Video width={width}, height={height}')

            # get inference state
            inference_state = get_inference_state(temp_dir)
            predictor.reset_state(inference_state)

            # Group prompts by 'obj_id' and sort them by 'frame_idx' in one step
            prompt_id_dict = defaultdict(list)
            [prompt_id_dict[prompt['obj_id']].append(prompt) for prompt in prompts]

            # Sort the prompts and extract the highest frame index for each object ID
            highest_frames = [sorted(prompts, key=lambda x: x['frame_idx'])[-1]['frame_idx'] for prompts in
                              prompt_id_dict.values() if prompts]

            # Get the minimum value of the highest frame indices
            prompt_idx = min(highest_frames) if highest_frames else None

            for prompt in prompts:

                frame_idx = prompt['frame_idx'] - first_frame_idx
                # sam 2 not predict other frame if are present prompts after the frame: the prompt must be set in the same frame for each object
                if frame_idx > prompt_idx:
                    logger.warning(f'Prompt frame index {frame_idx} is out of bounds')
                    continue


                if PROMPT_TYPE == 'point':
                    # multiply points by the frame size
                    prompt['points'][:, 0] *= width
                    prompt['points'][:, 1] *= height
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_ids[prompt['obj_id']],
                        points=prompt['points'],
                        labels=prompt['labels']
                    )
                elif PROMPT_TYPE == 'box':
                    # multiply points by the frame size
                    prompt['points'][0] *= width
                    prompt['points'][1] *= height
                    prompt['points'][2] *= width
                    prompt['points'][3] *= height
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_ids[prompt['obj_id']],
                        box=prompt['points'],
                    )
            if DEBUG:
              debug_dir = './debug-frames'
              os.makedirs(debug_dir, exist_ok=True)

            sequences = dict()
            logger.info(f'Propagating in video from frame {last_frame_idx} to {last_frame_idx + frames_to_track}')
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=last_frame_idx,
                max_frame_num_to_track=frames_to_track
            ):
                real_frame_idx = out_frame_idx + first_frame_idx
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                    if DEBUG:

                      # to debug, save the mask as an image
                      self.dump_image_with_mask(frames[out_frame_idx][1], mask, f'{debug_dir}/{out_frame_idx:05d}_{out_obj_id}.jpg', obj_id=out_obj_id, random_color=True)

                    bbox = self.convert_mask_to_bbox(mask)
                    if bbox:
                        obj_id = next((k for k, v in obj_ids.items() if v == out_obj_id), None)
                        sequences[obj_id] = sequences.get(obj_id, [])
                        sequences[obj_id].append({
                            'frame': real_frame_idx + 1,
                            # 'x': bbox['x'] / width * 100,
                            # 'y': bbox['y'] / height * 100,
                            # 'width': bbox['width'] / width * 100,
                            # 'height': bbox['height'] / height * 100,
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'enabled': True,
                            'rotation': 0,
                            'time': out_frame_idx / fps
                        })
            result = []
            for obj_id in all_obj_ids:
                # find the context to use by searching on drafts by obj_id
                context_result_sequence = next((ctx['value']['sequence'] for ctx in drafts["result"] if ctx['id'] == obj_id), [])
                # take the old sequence only for the frames before the first frame of the new sequence
                # and after the last frame of the new sequence
                new_sequence = [s for s in context_result_sequence if s['frame'] < sequences[obj_id][0]['frame']] + \
                               sequences[obj_id] + \
                               [s for s in context_result_sequence if s['frame'] >= sequences[obj_id][-1]['frame']]
                # take the old labels: take from context if present, otherwise from drafts
                labels = next((ctx['value'].get('labels', None) for ctx in context["result"] if ctx['id'] == obj_id), None) or \
                         next((ctx['value'].get('labels', None) for ctx in drafts["result"] if ctx['id'] == obj_id), None)
                result.append({
                    'value': {
                        'framesCount': frames_count,
                        'duration': duration,
                        'sequence': new_sequence,
                        'labels': labels if labels else []
                    },
                    'from_name': 'box',
                    'to_name': 'video',
                    'type': 'videorectangle',
                    'origin': 'manual',
                    'id': obj_id
                })


            prediction = PredictionValue(
                result=result
            )
            logger.debug(f'Prediction: {prediction.model_dump()}')
            if DEBUG:
              with open('prediction.json', 'w') as f:
                  json.dump(prediction.model_dump(), f)

            if ANNOTATION_WORKAROUND:
                # this is a workaround to update the annotation in the Label Studio since using the model response shows all the objects with the same label
                # also if the label is different for each object
                client = LabelStudio(
                    api_key="e89564826fd186964b1044cf6d13948fda91db09",
                )
                if len(tasks[0]['annotations']) == 0:
                    logger.debug('Creating new annotation')
                    ann = client.annotations.create(
                        id=task_id,
                        result=result,
                        task=tasks[0]['id'],
                        project=tasks[0]['project']
                    )
                    client.annotations.get(id=ann.id)
                else:
                    logger.debug(f'Updating annotation: {tasks[0]["annotations"][0]["id"]}')
                    ann = client.annotations.update(
                        id=tasks[0]['annotations'][0]['id'],
                        result=result,
                        task=task_id,
                        project=tasks[0]['project']
                    ) # perche se non lo faccio nella UI mette tutti gli oggetti con la stessa label! sempre!
                # convert annotation to draft making POST request to http://<IP>/api/annotations/{id}/convert-to-draft
                url = f'{os.getenv("LABEL_STUDIO_URL")}/api/annotations/{ann.id}/convert-to-draft'
                headers = {
                    'Authorization': f'Token {os.getenv("LABEL_STUDIO_API_KEY")}'
                }
                response = requests.post(url, headers=headers)
            # raise NotImplementedError('Stop here')
            return ModelResponse(predictions=[prediction])
