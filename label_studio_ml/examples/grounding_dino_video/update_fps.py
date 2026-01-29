"""CLI utility to synchronize FPS values for Label Studio video tasks."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import cv2
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from tqdm import tqdm

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Update Label Studio tasks with accurate video FPS values",
    )
    parser.add_argument(
        "--ls-url",
        type=str,
        default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio base URL",
    )
    parser.add_argument(
        "--ls-api-key",
        type=str,
        default=os.getenv("LABEL_STUDIO_API_KEY", "your_api_key"),
        help="Label Studio API key",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Label Studio project ID used to infer the video data key",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Path to JSON file or comma-separated list of task IDs. "
            "The JSON file can contain full task objects or bare IDs. "
            "If not provided, --project must be specified to fetch all tasks."
        ),
    )
    parser.add_argument(
        "--data-key",
        type=str,
        default=None,
        help="Explicit key in task['data'] that stores the video path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing task['data']['fps'] values",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity",
    )
    return parser.parse_args(args=argv)


def configure_logging(quiet: bool) -> None:
    """Initialize logging configuration respecting the quiet flag."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if not quiet else logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    else:
        logging.getLogger().setLevel(logging.WARNING if quiet else logging.INFO)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entry point for the CLI script."""
    args = parse_args(argv)
    configure_logging(args.quiet)

    os.environ.setdefault("LABEL_STUDIO_URL", args.ls_url)
    os.environ.setdefault("LABEL_STUDIO_API_KEY", args.ls_api_key)

    LOGGER.info("Connecting to Label Studio at %s", args.ls_url)
    ls = LabelStudio(base_url=args.ls_url, api_key=args.ls_api_key)

    data_key = args.data_key
    if not data_key and args.project:
        data_key = infer_video_data_key(ls, args.project)
        if data_key:
            LOGGER.info("Inferred video data key '%s' from project %s", data_key, args.project)
        else:
            LOGGER.warning(
                "Failed to infer video data key from project %s; will try heuristic fallback",
                args.project,
            )

    if not args.tasks and not args.project:
        LOGGER.error("Either --tasks or --project must be provided")
        return 1

    tasks = prepare_tasks(ls, args.tasks, args.project)
    if not tasks:
        LOGGER.error("No tasks to process")
        return 1

    processed = 0
    updated = 0

    for task in tqdm(tasks, desc="Update FPS", unit="task"):
        task_id = task.get("id")
        task_data = task.get("data")
        if not task_id or not isinstance(task_data, dict):
            LOGGER.warning("Skipping invalid task payload: %s", task)
            continue

        task_data_key = data_key or detect_video_key(task_data)
        if not task_data_key:
            LOGGER.warning("Task %s has no detectable video key", task_id)
            continue

        try:
            video_path = resolve_video_path(task, args.ls_url, task_data_key)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to resolve path for task %s: %s", task_id, exc)
            continue

        fps = extract_fps(video_path)
        if fps is None:
            LOGGER.warning("Could not determine FPS for task %s (path: %s)", task_id, video_path)
            continue

        if update_task_fps(ls, task_id, task_data, fps, args.overwrite):
            updated += 1
        processed += 1

    LOGGER.info("Processed %s tasks, updated %s", processed, updated)
    return 0 if processed else 1


def prepare_tasks(ls: LabelStudio, tasks_arg: Optional[str], project_id: Optional[str] = None) -> List[Dict]:
    """Load tasks either from a file, by fetching IDs, or by fetching all from a project."""
    if not tasks_arg:
        if not project_id:
            raise ValueError("Either tasks_arg or project_id must be provided")

        LOGGER.info("Fetching all tasks from project %s", project_id)
        tasks = []
        for task in tqdm(ls.tasks.list(project=project_id), desc="Fetch project tasks", unit="task"):
            tasks.append({"id": task.id, "data": task.data})
        return tasks

    if os.path.exists(tasks_arg):
        with open(tasks_arg, "r", encoding="utf-8") as handle:
            tasks_raw = json.load(handle)
    else:
        tasks_raw = [int(task_id) for task_id in tasks_arg.split(",") if task_id.strip()]

    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("Tasks input must be a non-empty list")

    first = tasks_raw[0]
    if isinstance(first, dict):
        for task in tasks_raw:
            if not isinstance(task, dict) or "id" not in task or "data" not in task:
                raise ValueError("Task dictionary entries must include 'id' and 'data'")
        return tasks_raw

    if not all(isinstance(item, int) for item in tasks_raw):
        raise ValueError("Task list must contain either dicts or integers")

    LOGGER.info("Fetching %d tasks from Label Studio", len(tasks_raw))
    tasks: List[Dict] = []
    for task_id in tqdm(tasks_raw, desc="Fetch tasks", unit="task"):
        task = ls.tasks.get(task_id)
        tasks.append({"id": task_id, "data": task.data})
    return tasks


def infer_video_data_key(ls: LabelStudio, project_id: str) -> Optional[str]:
    """Inspect project labeling config to discover the video object's data key."""
    try:
        project = ls.projects.get(id=project_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to fetch project %s: %s", project_id, exc)
        return None

    try:
        label_interface = LabelInterface(config=project.label_config)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unable to parse label config for project %s: %s", project_id, exc)
        return None

    for control in label_interface.objects:
        if control.tag == "Video" and control.value_name:
            return control.value_name
    return None


def detect_video_key(task_data: Dict) -> Optional[str]:
    """Guess the key storing video reference using a simple heuristic."""
    for key, value in task_data.items():
        if isinstance(value, str) and is_video_like(value):
            return key
    return None


def resolve_video_path(task: Dict, base_url: str, data_key: str) -> str:
    """Resolve a task's video path to a local file accessible by OpenCV."""
    data = task.get("data", {})
    if data_key not in data:
        raise ValueError(f"Task data does not contain key '{data_key}'")

    task_path = data[data_key]
    if not isinstance(task_path, str):
        raise ValueError(f"Expected string path for key '{data_key}', got {type(task_path)}")

    if not task_path.startswith("http") and task_path.startswith("/"):
        host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL") or base_url
        if host:
            task_path = urljoin(host.rstrip("/"), task_path)
        else:
            LOGGER.debug(
                "Relative task path %s found but no Label Studio host URL configured",
                task_path,
            )

    download_source = task_path
    path = task_path if os.path.exists(task_path) else get_local_path(task_path, task_id=task.get("id"))

    if os.path.exists(path):
        path = ensure_extension(path, download_source)

    LOGGER.debug("Resolved task %s path %s", task.get("id"), path)
    return path


def ensure_extension(path: str, download_source: str) -> str:
    """Append an extension to a downloaded file when necessary."""
    suffix = Path(path).suffix
    if suffix:
        return path

    parsed = urlparse(download_source)
    candidate_suffix = Path(parsed.path).suffix
    if not candidate_suffix:
        query = parse_qs(parsed.query)
        fileuri = (query.get("fileuri") or [None])[0]
        if fileuri:
            padding = (-len(fileuri)) % 4
            fileuri_padded = fileuri + ("=" * padding)
            try:
                decoded = base64.urlsafe_b64decode(fileuri_padded).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                decoded = ""
            candidate_suffix = Path(decoded).suffix

    if not candidate_suffix:
        return path

    candidate_path = f"{path}{candidate_suffix}"
    if not os.path.exists(candidate_path):
        try:
            os.symlink(path, candidate_path)
        except OSError:
            shutil.copyfile(path, candidate_path)
    return candidate_path


def is_video_like(value: str) -> bool:
    """Check whether a string resembles a video path or URL."""
    lowered = value.lower()
    if lowered.startswith(("http://", "https://")):
        return True
    if lowered.startswith("/"):
        return True
    return any(lowered.endswith(ext) for ext in VIDEO_EXTENSIONS)


def extract_fps(path: str) -> Optional[float]:
    """Read FPS metadata using OpenCV."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if fps <= 0:
        return None
    return round(fps, 6)


def update_task_fps(
    ls: LabelStudio,
    task_id: int,
    task_data: Dict,
    fps: float,
    overwrite: bool,
    epsilon: float = 1e-6,
) -> bool:
    """Update task data with FPS if needed."""
    current = task_data.get("fps")
    if current is not None and not overwrite:
        try:
            current_value = float(current)
        except (TypeError, ValueError):
            current_value = None
        if current_value is not None and abs(current_value - fps) <= epsilon:
            LOGGER.debug("Task %s already has FPS %s", task_id, current_value)
            return False
        LOGGER.info(
            "Task %s already has fps=%s; skipping because overwrite is disabled",
            task_id,
            current,
        )
        return False

    task_data["fps"] = fps
    try:
        ls.tasks.update(task_id, data=task_data)
        LOGGER.info("Updated task %s with fps=%.6f", task_id, fps)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to update task %s: %s", task_id, exc)
        return False


if __name__ == "__main__":
    sys.exit(main())
