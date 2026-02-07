"""Utilities and CLI for deleting a single annotation or prediction in Label Studio.

The main entrypoint is a small CLI that:
- Connects to Label Studio via the Python SDK
- Validates that the referenced task exists (and logs project/task info)
- Deletes either an annotation or a prediction by ID

Usage examples:

    python delete_annotation_or_prediction.py \
        --ls-url https://app.heartex.com \
        --ls-api-key "$LABEL_STUDIO_API_KEY" \
        --project 198563 \
        --task 226454005 \
        --annotation 78741264

    python delete_annotation_or_prediction.py \
        --ls-url https://app.heartex.com \
        --ls-api-key "$LABEL_STUDIO_API_KEY" \
        --project 198563 \
        --task 226454005 \
        --prediction 123456789

The same script can be called inside your Docker container similarly to cli.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class DeleteCLIError(Exception):
    """Custom error type for the delete CLI."""


def _build_ls_client(ls_url: str, ls_api_key: str):
    """Create a Label Studio SDK client.

    Performs basic validation of the API key and sets a reasonable timeout.
    """
    if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
        raise DeleteCLIError(
            "LABEL_STUDIO_API_KEY is required. "
            "Provide it via --ls-api-key or the LABEL_STUDIO_API_KEY env var."
        )

    from label_studio_sdk.client import LabelStudio

    logger.info("Connecting to Label Studio at %s", ls_url)
    client = LabelStudio(base_url=ls_url, api_key=ls_api_key, timeout=600)
    logger.info("Connected to Label Studio")
    return client


def _fetch_task(ls, project_id: int, task_id: int) -> Any:
    """Fetch and minimally validate a task from Label Studio.

    Project ID is currently used for logging and sanity checks only.
    """
    logger.info("Fetching task %s from project %s", task_id, project_id)
    task_obj = ls.tasks.get(task_id)
    if not task_obj:
        raise DeleteCLIError(f"Task {task_id} not found")

    logger.info("Task fetched: %s", getattr(task_obj, "id", task_id))

    # If the task object exposes a project field, try to log/validate it
    task_project = getattr(task_obj, "project", None)
    if task_project is not None and task_project != project_id:
        logger.warning(
            "Task %s belongs to project %s (not %s)",
            getattr(task_obj, "id", task_id),
            task_project,
            project_id,
        )

    return task_obj


def delete_annotation_or_prediction(
    ls,
    project_id: int,
    task_id: int,
    annotation_id: Optional[Union[int, Sequence[int]]] = None,
    prediction_id: Optional[Union[int, Sequence[int]]] = None,
) -> None:
    """Delete either a single annotation or prediction by ID.

    Exactly one of `annotation_id` or `prediction_id` must be provided.
    Project and task IDs are used for logging and basic sanity checks.
    """
    ann_ids: Sequence[int] = []
    pred_ids: Sequence[int] = []

    if annotation_id is not None:
        if isinstance(annotation_id, int):
            ann_ids = [annotation_id]
        else:
            ann_ids = [v for v in annotation_id if isinstance(v, int)]

    if prediction_id is not None:
        if isinstance(prediction_id, int):
            pred_ids = [prediction_id]
        else:
            pred_ids = [v for v in prediction_id if isinstance(v, int)]

    if (not ann_ids and not pred_ids) or (ann_ids and pred_ids):
        raise DeleteCLIError("Provide exactly one of annotation_id or prediction_id (one or many IDs)")

    # Ensure the task exists (and log project/task info)
    _fetch_task(ls, project_id=project_id, task_id=task_id)

    if ann_ids:
        for ann_id in ann_ids:
            logger.info(
                "Preparing to delete annotation %s (project=%s, task=%s)",
                ann_id,
                project_id,
                task_id,
            )
            obj = ls.annotations.get(id=ann_id)
            if not obj:
                raise DeleteCLIError(f"Annotation {ann_id} not found")

            obj_task = getattr(obj, "task", None) or getattr(obj, "task_id", None)
            if obj_task is not None and obj_task != task_id:
                logger.warning(
                    "Annotation %s is attached to task %s (not %s)",
                    ann_id,
                    obj_task,
                    task_id,
                )

            obj_project = getattr(obj, "project", None)
            if obj_project is not None and obj_project != project_id:
                logger.warning(
                    "Annotation %s is attached to project %s (not %s)",
                    ann_id,
                    obj_project,
                    project_id,
                )

            logger.info("Deleting annotation %s...", ann_id)
            ls.annotations.delete(id=ann_id)
            logger.info("Annotation %s deleted", ann_id)
        return

    for pred_id in pred_ids:
        logger.info(
            "Preparing to delete prediction %s (project=%s, task=%s)",
            pred_id,
            project_id,
            task_id,
        )

        obj = ls.predictions.get(id=pred_id)
        if not obj:
            raise DeleteCLIError(f"Prediction {pred_id} not found")

        obj_task = getattr(obj, "task", None) or getattr(obj, "task_id", None)
        if obj_task is not None and obj_task != task_id:
            logger.warning(
                "Prediction %s is attached to task %s (not %s)",
                pred_id,
                obj_task,
                task_id,
            )

        obj_project = getattr(obj, "project", None)
        if obj_project is not None and obj_project != project_id:
            logger.warning(
                "Prediction %s is attached to project %s (not %s)",
                pred_id,
                obj_project,
                project_id,
            )

        logger.info("Deleting prediction %s...", pred_id)
        ls.predictions.delete(id=pred_id)
        logger.info("Prediction %s deleted", pred_id)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the delete utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Delete a single Label Studio annotation or prediction by ID, "
            "given project and task identifiers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python delete_annotation_or_prediction.py --ls-url https://app.heartex.com \\\n"
            "    --ls-api-key YOUR_KEY --project 123 --task 456 --annotation 789\n\n"
            "  python delete_annotation_or_prediction.py --ls-url https://app.heartex.com \\\n"
            "    --ls-api-key YOUR_KEY --project 123 --task 456 --prediction 555\n"
        ),
    )

    parser.add_argument(
        "--ls-url",
        required=True,
        help="Label Studio URL (e.g., https://app.heartex.com)",
    )
    parser.add_argument(
        "--ls-api-key",
        required=True,
        help="Label Studio API key",
    )
    parser.add_argument(
        "--project",
        type=int,
        required=True,
        help="Project ID (used for logging/validation)",
    )
    parser.add_argument(
        "--task",
        type=int,
        required=True,
        help="Task ID associated with the annotation/prediction",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--annotation",
        type=int,
        nargs="+",
        help="Annotation ID to delete",
    )
    source_group.add_argument(
        "--prediction",
        type=int,
        nargs="+",
        help="Prediction ID to delete",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for deleting annotations or predictions."""
    args = _parse_args()

    # Set global log level according to CLI argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("🗑  DELETE ANNOTATION/PREDICTION CLI STARTED")
    logger.info("=" * 80)
    logger.info("📋 Parameters:")
    logger.info("   • Label Studio URL: %s", args.ls_url)
    logger.info("   • Project ID: %s", args.project)
    logger.info("   • Task ID: %s", args.task)
    if args.annotation is not None:
        logger.info("   • Target: annotation %s", args.annotation)
    if args.prediction is not None:
        logger.info("   • Target: prediction %s", args.prediction)
    logger.info("=" * 80)

    exit_code = 0

    try:
        ls = _build_ls_client(args.ls_url, args.ls_api_key)

        delete_annotation_or_prediction(
            ls,
            project_id=args.project,
            task_id=args.task,
            annotation_id=args.annotation,
            prediction_id=args.prediction,
        )

        logger.info("=" * 80)
        logger.info("✅ DELETE CLI EXECUTION SUCCESSFUL")
        logger.info("=" * 80)

    except DeleteCLIError as e:
        logger.error("❌ Delete CLI error: %s", e)
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        exit_code = 130
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.error("❌ Unexpected error: %s", e, exc_info=True)
        exit_code = 1
    finally:
        if exit_code != 0:
            logger.info("=" * 80)
            logger.info("❌ DELETE CLI EXECUTION FAILED (exit code: %s)", exit_code)
            logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
