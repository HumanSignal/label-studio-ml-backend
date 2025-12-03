import os
import logging
import json

import httpx
from tqdm import tqdm
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from model import YOLO
from label_studio_sdk.client import LabelStudio
from label_studio_ml.response import ModelResponse
from tracking_presets import (
    TRACKING_LAYERS,
    apply_preset,
    list_presets,
    describe_preset,
)

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "your_api_key")
PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def arg_parser():
    epilog = (
        "Composable Tracking Presets:\n"
        "  Layers can be combined with '+' to address multiple concerns:\n"
        "\n"
        "  PLATFORM:  uav, ugv, handheld, fixed\n"
        "  SCENE:     crowded, sparse, cluttered\n"
        "  MOTION:    fast_motion, slow_motion, erratic\n"
        "  DURATION:  long_video, short_clip\n"
        "  MODALITY:  thermal, lowlight, hdr\n"
        "  QUALITY:   high_precision, high_recall\n"
        "\n"
        "Examples:\n"
        "  python cli.py --preset uav --project 123 --tasks 456\n"
        "  python cli.py --preset uav+fast_motion+long_video --project 123 --tasks 456\n"
        "  python cli.py --preset thermal+crowded+high_precision --project 123 --tasks 456\n"
        "\n"
        "Use --list-presets to see all layers, --describe-preset to see computed values.\n"
    )
    parser = ArgumentParser(
        description="Grounding DINO Video client for Label Studio ML Backend",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "--ls-url", type=str, default=LABEL_STUDIO_URL, help="Label Studio URL"
    )
    parser.add_argument(
        "--ls-api-key",
        type=str,
        default=LABEL_STUDIO_API_KEY,
        help="Label Studio API Key",
    )
    parser.add_argument(
        "--project", type=str, default="1", help="Label Studio Project ID"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="tasks.json",
        help="Path to tasks JSON file with list of ids or task datas. Example: tasks.json\n"
             "String with ids separated by comma: if you provide task ids, "
             "task data will be downloaded automatically from the Label Studio instance. Example: 1,2,3",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save annotated frames with bounding boxes for debugging (default: ./output_frames)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frames with bounding boxes to output directory",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames (for quick testing)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=os.getenv("TRACKING_PRESET"),
        help=(
            "Tracking preset(s) for detection/tracking parameters. "
            "Combine multiple layers with '+' (e.g., 'uav+fast_motion+long_video'). "
            "Defaults to env TRACKING_PRESET."
        ),
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available tracking layers and exit.",
    )
    parser.add_argument(
        "--describe-preset",
        type=str,
        metavar="PRESET",
        help="Show computed parameter values for a preset and exit.",
    )
    return parser.parse_args()


class LabelStudioMLPredictor:
    def __init__(self, ls_url, ls_api_key):
        # Validate API key
        if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
            raise ValueError(
                "LABEL_STUDIO_API_KEY is required. Please set it via environment variable or --ls-api-key argument."
            )

        # Set environment variables for SDK internal functions (like get_local_path)
        os.environ.setdefault("LABEL_STUDIO_URL", ls_url)
        os.environ.setdefault("LABEL_STUDIO_API_KEY", ls_api_key)

        self.ls = LabelStudio(base_url=ls_url, api_key=ls_api_key)
        logger.info(f"Successfully connected to Label Studio: {ls_url}")

    def run(self, project, tasks, output_dir=None, save_frames=False, max_frames=None):
        # initialize Label Studio SDK client
        ls = self.ls
        project = ls.projects.get(id=project)
        logger.info(f"Project is retrieved: {project.id}")

        tasks = self.prepare_tasks(ls, tasks)

        # load Grounding DINO video model
        # TODO: use get_all_classes_inherited_LabelStudioMLBase to detect model classes
        model = YOLO(project_id=str(project.id), label_config=project.label_config)
        logger.info("Grounding DINO ML backend is created")

        # Setup output directory if saving frames
        if save_frames and not output_dir:
            output_dir = "./output_frames"
        
        if save_frames:
            logger.info(f"Annotated frames will be saved to: {output_dir}")

        # predict and send prediction to Label Studio
        for task in tqdm(tasks, desc="Predict tasks"):
            fps_synced = False
            response = model.predict(
                [task],
                output_dir=output_dir,
                save_frames=save_frames,
                max_frames=max_frames,
            )
            predictions = self.postprocess_response(model, response, task)

            # send predictions to Label Studio
            for prediction in predictions:
                score = prediction.get("score", 0)
                logger.info(
                    "Submitting prediction for task %s with score=%.4f",
                    task.get("id"),
                    float(score) if isinstance(score, (int, float)) else score,
                )
                try:
                    ls.predictions.create(
                        task=task["id"],
                        score=score,
                        model_version=prediction.get("model_version", "none"),
                        result=prediction["result"],
                    )
                except httpx.ReadTimeout as exc:
                    logger.warning(
                        "Prediction likely created but Label Studio timed out while acknowledging (non-fatal timeout): %s",
                        exc,
                    )
                except httpx.HTTPStatusError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 504:
                        logger.warning(
                            "Label Studio returned HTTP 504 Gateway Timeout while saving prediction; treating as non-fatal. Response: %s",
                            exc,
                        )
                    else:
                        raise

            if not fps_synced:
                fps_synced = self._update_task_fps_if_needed(task)

        logger.info("Model predictions are done!")
        if save_frames:
            logger.info(f"Annotated frames saved to: {output_dir}")

    def _update_task_fps_if_needed(self, task):
        if not isinstance(task, dict):
            return False

        task_id = task.get("id")
        task_data = task.get("data")
        if not task_id or not isinstance(task_data, dict):
            return False

        fps = task_data.get("fps")
        if fps is None:
            return False

        try:
            self.ls.tasks.update(task_id, data=task_data)
            logger.info("Updated task %s with fps=%.4f", task_id, fps)
            return True
        except Exception as exc:
            logger.warning("Failed to update task %s with fps: %s", task_id, exc)
            return False

    @staticmethod
    def postprocess_response(model, response, task):
        if response is None:
            logger.warning(f"No predictions for task: {task}")
            return None

        # model returned ModelResponse
        if isinstance(response, ModelResponse):
            # check model version
            if not response.has_model_version():
                if model.model_version:
                    response.set_version(str(model.model_version))
            else:
                response.update_predictions_version()
            response = response.model_dump()
            predictions = response.get("predictions")
        # model returned list of dicts with predictions (old format)
        elif isinstance(response, list):
            predictions = response
        else:
            logger.error("No predictions generated by model")
            return None

        return predictions

    @staticmethod
    def prepare_tasks(ls, tasks):
        # get tasks
        if os.path.exists(tasks):
            with open(tasks) as f:
                tasks = json.load(f)
        else:
            tasks = tasks.split(",")
            tasks = [int(task) for task in tasks]
        assert isinstance(tasks, list), "Tasks should be a list"
        assert len(tasks) > 0, "'Task list can't be empty"
        logger.info(f"Detected {len(tasks)} tasks")
        # check task data
        if isinstance(tasks[0], dict):
            if "data" not in tasks[0] or "id" not in tasks[0]:
                raise ValueError("'data' and 'id' must be presented in all tasks")
        elif isinstance(tasks[0], int):
            # load tasks from Label Studio instance using SDK
            logger.info("Task loading from Label Studio instance ...")
            tasks = [
                {"id": task_id, "data": ls.tasks.get(task_id).data}
                for task_id in tqdm(tasks)
            ]
            logger.info("Task loading finished")
        else:
            raise ValueError(
                "Unknown task format: "
                "tasks should be a list of dicts (task data) or a list of task ids"
            )
        return tasks


if __name__ == "__main__":
    args = arg_parser()

    # Handle --list-presets
    if args.list_presets:
        print(list_presets())
        exit(0)

    # Handle --describe-preset
    if args.describe_preset:
        print(describe_preset(args.describe_preset))
        exit(0)

    # Apply tracking preset if specified (must happen before model init)
    if args.preset:
        preset = apply_preset(args.preset)
        logger.info(
            "Using tracking preset '%s': %s",
            preset.name,
            preset.description,
        )
        # Show computed values for transparency
        logger.info(
            "Computed parameters: box=%.2f, text=%.2f, score=%.2f, "
            "activation=%.2f, buffer=%d, match=%.2f, consecutive=%d",
            preset.box_threshold,
            preset.text_threshold,
            preset.model_score_threshold,
            preset.track_activation_threshold,
            preset.lost_track_buffer,
            preset.minimum_matching_threshold,
            preset.minimum_consecutive_frames,
        )

    predictor = LabelStudioMLPredictor(args.ls_url, args.ls_api_key)
    predictor.run(
        args.project,
        args.tasks,
        output_dir=args.output_dir,
        save_frames=args.save_frames,
        max_frames=args.max_frames,
    )
