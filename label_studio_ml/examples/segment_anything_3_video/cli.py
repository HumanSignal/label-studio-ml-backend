#!/usr/bin/env python
"""
CLI for SAM3 Video Tracking - Process Label Studio tasks with existing annotations

This CLI allows batch processing of video tracking tasks by:
1. Fetching task and annotation data from Label Studio
2. Running SAM3 tracking on keyframes
3. Uploading predictions back to Label Studio

Migrated from SAM2 to SAM3 via HuggingFace Transformers.

Usage:
    python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY \
                  --project 123 --task 456 --annotation 789
"""

import os
import sys
import argparse
import logging
import signal
import time
import threading
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


class CLIError(Exception):
    """Custom CLI error for graceful failure"""
    pass


def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.warning(f'\nReceived signal {signal_name}, shutting down gracefully...')
        sys.exit(130)  # Standard exit code for SIGINT

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_environment():
    """Validate required environment variables and system requirements"""
    logger.info('Validating environment...')

    # Check CUDA availability
    try:
        import torch
        if os.getenv('DEVICE', 'cuda') == 'cuda':
            if not torch.cuda.is_available():
                logger.error('CUDA not available but DEVICE=cuda')
                raise CLIError('GPU required but not available')
            logger.info(f'GPU available: {torch.cuda.get_device_name(0)}')
    except ImportError:
        logger.error('PyTorch not installed')
        raise CLIError('PyTorch is required')

    # Check SAM3 installation via HuggingFace transformers
    hints = os.getenv('HINTS', 'false').lower() == 'true'
    try:
        if hints:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
            logger.info('SAM3 VideoModel (PCS) available via transformers')
        else:
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
            logger.info('SAM3 TrackerVideoModel available via transformers')
    except ImportError:
        logger.error('SAM3 not available in transformers')
        raise CLIError('SAM3 is required. Install transformers from git.')

    logger.info('Environment validation complete')


def fetch_task_data(ls, project_id: int, task_id: int, annotation_id: int):
    """Fetch task and annotation data from Label Studio with timeout"""
    logger.info(f'Fetching task {task_id} from project {project_id}...')

    start_time = time.time()
    timeout = 60  # 60 second timeout for API calls

    try:
        # Fetch task
        task_obj = ls.tasks.get(task_id)
        task = {"id": task_obj.id, "data": task_obj.data}
        if not task:
            raise CLIError(f'Task {task_id} not found')
        logger.info(f'Task fetched: {task.get("id")}')

        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError(f'Task fetch exceeded {timeout}s timeout')

        # Fetch annotation
        logger.info(f'Fetching annotation {annotation_id}...')
        annotation = ls.annotations.get(id=annotation_id)

        if not annotation:
            raise CLIError(f'Annotation {annotation_id} not found')

        # Convert annotation to dict format
        annotation_dict = {
            "id": annotation.id,
            "result": annotation.result
        }

        logger.info(f'Annotation fetched: {annotation_dict.get("id")} with {len(annotation_dict.get("result", []))} regions')

        # Validate annotation has regions
        if not annotation_dict.get('result'):
            raise CLIError(f'Annotation {annotation_id} has no keyframe regions')

        return task, annotation_dict

    except TimeoutError:
        logger.error(f'API request timed out after {timeout}s')
        raise
    except Exception as e:
        logger.error(f'Failed to fetch data: {e}')
        raise CLIError(f'API error: {e}')


def upload_prediction(ls, task_id: int, prediction_data: dict):
    """Upload prediction to Label Studio with progress feedback and retry mechanism"""
    logger.info(f'Uploading prediction for task {task_id}...')

    # Estimate upload size for progress feedback
    import json
    prediction_json = json.dumps(prediction_data)
    upload_size_mb = len(prediction_json.encode('utf-8')) / (1024 * 1024)
    logger.info(f'Upload size: {upload_size_mb:.2f}MB')

    # Heartbeat logging during upload - use threading.Event for thread-safe control
    upload_start_time = time.time()
    heartbeat_stop = threading.Event()

    def upload_heartbeat():
        """Log upload progress every 10 seconds"""
        while not heartbeat_stop.is_set():
            elapsed = time.time() - upload_start_time
            logger.info(f'Upload in progress... ({elapsed:.0f}s elapsed)')
            heartbeat_stop.wait(10)  # Sleep with interruptibility

    # Start heartbeat thread
    heartbeat_thread = threading.Thread(target=upload_heartbeat, daemon=True)
    heartbeat_thread.start()

    # Retry mechanism with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f'Upload attempt {attempt + 1}/{max_retries}...')

            result = ls.predictions.create(
                task=task_id,
                score=prediction_data.get('score', 0),
                model_version=prediction_data.get('model_version', 'sam3'),
                result=prediction_data.get('result', [])
            )

            # Stop heartbeat on success
            heartbeat_stop.set()
            upload_elapsed = time.time() - upload_start_time
            logger.info(f'Upload completed in {upload_elapsed:.1f}s!')

            return result

        except Exception as e:
            # Check if this is a 504 Gateway Timeout - data was sent successfully
            error_str = str(e)
            if '504' in error_str and ('Gateway Time-out' in error_str or 'Gateway Timeout' in error_str):
                heartbeat_stop.set()  # Stop heartbeat - we're done
                logger.warning(f'Got 504 Gateway Timeout. Data was sent successfully; treating as success.')
                return None  # Success, but no response object

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f'Upload failed (attempt {attempt + 1}): {e}')
                logger.info(f'Retrying in {wait_time}s...')
                time.sleep(wait_time)
            else:
                heartbeat_stop.set()  # Stop heartbeat - we're done
                logger.error(f'Upload failed after {max_retries} attempts: {e}')
                raise CLIError(f'Upload error: {e}')


def main():
    """Main CLI execution"""
    parser = argparse.ArgumentParser(
        description='SAM3 Video Tracking CLI - Process Label Studio tasks with keyframe annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracking (uses env defaults from docker-compose.yml)
  python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY --project 123 --task 456 --annotation 789

  # Limit to 500 frames with debug logging
  python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY --project 123 --task 456 --annotation 789 --max-frames 500 --log-level DEBUG

  # Text-detection mode (Sam3VideoModel with text prompts)
  python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY --project 123 --task 456 --annotation 789 --hints true --prompt-text "person"

  # Chunked batch mode with temporal downsampling
  python cli.py --ls-url https://app.heartex.com --ls-api-key YOUR_KEY --project 123 --task 456 --annotation 789 --processing-mode chunked_batch --track-fps 5
        """
    )

    # -- Required: Label Studio connection --
    parser.add_argument('--ls-url', required=True, help='Label Studio URL (e.g., https://app.heartex.com)')
    parser.add_argument('--ls-api-key', required=True, help='Label Studio API key')
    parser.add_argument('--project', type=int, required=True, help='Project ID')
    parser.add_argument('--task', type=int, required=True, help='Task ID to process')
    parser.add_argument('--annotation', type=int, required=True, help='Annotation ID with keyframes')

    # -- Optional: model and tracking configuration --
    parser.add_argument('--device', default=None,
                       help='Compute device (default: env DEVICE or "cuda")')
    parser.add_argument('--hints', choices=['true', 'false'], default=None,
                       help='Text-detection mode: true=Sam3VideoModel, false=Sam3TrackerVideoModel '
                            '(default: env HINTS or "false")')
    parser.add_argument('--model-name', default=None,
                       help='HuggingFace model ID (default: env MODEL_NAME or "facebook/sam3")')
    parser.add_argument('--processing-mode', choices=['streaming', 'chunked_batch'], default=None,
                       help='Processing mode (default: env PROCESSING_MODE or "streaming")')
    parser.add_argument('--track-fps', type=float, default=None,
                       help='Target FPS for temporal downsampling, 0=no downsampling '
                            '(default: env TRACK_FPS or 0)')
    parser.add_argument('--prompt-text', default=None,
                       help='Text prompt for detection when --hints=true '
                            '(default: env PROMPT_TEXT or "person")')
    parser.add_argument('--max-frames', type=int, default=0,
                       help='Max frames to track, 0=unlimited '
                            '(default: env MAX_FRAMES_TO_TRACK or 0)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Setup signal handlers
    setup_signal_handlers()

    # ------------------------------------------------------------------
    # Set environment variables from CLI args BEFORE any model imports.
    # model.py reads env vars at module-import time, so these must be
    # set before ``from model import NewModel`` and validate_environment.
    # Only override when the user explicitly provided the flag; otherwise
    # the existing env (e.g. from docker-compose.yml) is preserved.
    # ------------------------------------------------------------------
    if args.device is not None:
        os.environ['DEVICE'] = args.device
    if args.hints is not None:
        os.environ['HINTS'] = args.hints
    if args.model_name is not None:
        os.environ['MODEL_NAME'] = args.model_name
    if args.processing_mode is not None:
        os.environ['PROCESSING_MODE'] = args.processing_mode
    if args.track_fps is not None:
        os.environ['TRACK_FPS'] = str(args.track_fps)
    if args.prompt_text is not None:
        os.environ['PROMPT_TEXT'] = args.prompt_text
    if args.max_frames > 0:
        os.environ['MAX_FRAMES_TO_TRACK'] = str(args.max_frames)

    # Log effective configuration (CLI override → env → default)
    logger.info('='*80)
    logger.info('SAM3 VIDEO CLI STARTED')
    logger.info('='*80)
    logger.info(f'Parameters:')
    logger.info(f'   Label Studio URL: {args.ls_url}')
    logger.info(f'   Project ID: {args.project}')
    logger.info(f'   Task ID: {args.task}')
    logger.info(f'   Annotation ID: {args.annotation}')
    logger.info(f'   Device: {os.getenv("DEVICE", "cuda")}')
    logger.info(f'   Hints: {os.getenv("HINTS", "false")}')
    logger.info(f'   Model: {os.getenv("MODEL_NAME", "facebook/sam3")}')
    logger.info(f'   Processing mode: {os.getenv("PROCESSING_MODE", "streaming")}')
    logger.info(f'   Track FPS: {os.getenv("TRACK_FPS", "0")}')
    logger.info(f'   Prompt text: {os.getenv("PROMPT_TEXT", "") or "(empty)"}')
    logger.info(f'   Max frames: {args.max_frames if args.max_frames > 0 else "unlimited"}')
    logger.info('='*80)

    exit_code = 0

    try:
        # Validate environment
        validate_environment()

        # Initialize Label Studio client
        logger.info('Connecting to Label Studio...')
        from label_studio_sdk.client import LabelStudio

        ls = LabelStudio(
            base_url=args.ls_url,
            api_key=args.ls_api_key,
            timeout=600  # 10 minute timeout for large uploads
        )
        logger.info('Connected to Label Studio')

        # Fetch task and annotation data
        task, annotation = fetch_task_data(ls, args.project, args.task, args.annotation)

        # Get label config from project
        logger.info(f'Fetching project configuration...')
        project = ls.projects.get(id=args.project)
        label_config = project.label_config or project.parsed_label_config

        if not label_config:
            raise CLIError('Could not fetch label config from project')

        logger.info('Label config fetched')

        # Initialize model
        logger.info('Initializing SAM3 model...')
        from model import NewModel

        model = NewModel(label_config=label_config)
        logger.info('Model initialized')

        # Prepare context from annotation
        context = {
            'result': annotation['result']
        }

        # Run prediction
        logger.info('Starting SAM3 tracking...')
        start_time = time.time()

        response = model.predict(tasks=[task], context=context)

        elapsed = time.time() - start_time
        logger.info(f'Tracking complete in {elapsed:.2f}s')

        # Extract prediction data
        if not response or not response.predictions:
            raise CLIError('Model returned no predictions')

        prediction = response.predictions[0]
        prediction_data = prediction.model_dump() if hasattr(prediction, 'model_dump') else prediction

        # Upload prediction
        upload_prediction(ls, args.task, prediction_data)

        logger.info('='*80)
        logger.info('CLI EXECUTION SUCCESSFUL')
        logger.info('='*80)

    except KeyboardInterrupt:
        logger.warning('\nInterrupted by user')
        exit_code = 130
    except CLIError as e:
        logger.error(f'CLI Error: {e}')
        exit_code = 1
    except TimeoutError as e:
        logger.error(f'Timeout Error: {e}')
        exit_code = 124  # Standard timeout exit code
    except Exception as e:
        logger.error(f'Unexpected error: {e}', exc_info=True)
        exit_code = 1
    finally:
        if exit_code != 0:
            logger.info('='*80)
            logger.info(f'CLI EXECUTION FAILED (exit code: {exit_code})')
            logger.info('='*80)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
