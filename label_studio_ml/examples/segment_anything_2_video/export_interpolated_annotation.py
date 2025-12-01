from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from label_studio_sdk.client import LabelStudio


class ExportCLIError(Exception):
    """Custom error type dedicated to the export helper script."""


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@dataclass
class ExportHandle:
    base_url: str
    session: requests.Session


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a single Label Studio annotation with interpolated frames enabled. "
            "The script creates an export snapshot, waits for completion, downloads the "
            "JSON payload, and stores only the requested annotation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (inside Docker):\n\n"
            "  docker compose exec segment_anything_2_video bash -lc '" "\n"
            "    export LABEL_STUDIO_HOST=https://app.heartex.com &&" "\n"
            "    export LABEL_STUDIO_URL=https://app.heartex.com &&" "\n"
            "    export LABEL_STUDIO_API_KEY=\"$LABEL_STUDIO_API_KEY\" &&" "\n"
            "    python /app/export_interpolated_annotation.py \\\n"
            "      --ls-url https://app.heartex.com \\\n"
            "      --ls-api-key \"$LABEL_STUDIO_API_KEY\" \\\n"
            "      --project 123 \\\n"
            "      --task 456 \\\n"
            "      --annotation 789 \\\n"
            "      --output /data/interpolated_annotation.json" "\n"
            "  '"
        ),
    )

    parser.add_argument("--ls-url", required=True, help="Label Studio URL (e.g., https://app.heartex.com)")
    parser.add_argument("--ls-api-key", required=True, help="Label Studio API key")
    parser.add_argument("--project", type=int, required=True, help="Project ID that owns the task")
    parser.add_argument("--task", type=int, required=True, help="Task ID that contains the annotation")
    parser.add_argument("--annotation", type=int, required=True, help="Annotation ID to download")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the filtered export JSON. Default is auto-generated in the current directory.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between export status checks (default: 5s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum seconds to wait for export completion (default: 300s)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def _build_ls_client(ls_url: str, ls_api_key: str) -> LabelStudio:
    if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
        raise ExportCLIError(
            "LABEL_STUDIO_API_KEY is required. Provide it via --ls-api-key or LABEL_STUDIO_API_KEY env var."
        )

    os.environ.setdefault("LABEL_STUDIO_URL", ls_url)
    os.environ.setdefault("LABEL_STUDIO_API_KEY", ls_api_key)

    logger.info("Connecting to Label Studio at %s", ls_url)
    client = LabelStudio(base_url=ls_url, api_key=ls_api_key, timeout=600)
    logger.info("Connected to Label Studio")
    return client


def _fetch_task(ls: LabelStudio, project_id: int, task_id: int) -> Dict[str, Any]:
    logger.info("Fetching task %s from project %s", task_id, project_id)
    task_obj = ls.tasks.get(task_id)
    if not task_obj:
        raise ExportCLIError(f"Task {task_id} not found")

    task_project = getattr(task_obj, "project", None)
    if task_project is not None and task_project != project_id:
        logger.warning(
            "Task %s belongs to project %s (not %s)",
            getattr(task_obj, "id", task_id),
            task_project,
            project_id,
        )

    task = {"id": task_obj.id, "data": getattr(task_obj, "data", {})}
    logger.info("Task fetched: %s", task.get("id"))
    return task


def _fetch_annotation(ls: LabelStudio, annotation_id: int) -> Any:
    logger.info("Fetching annotation %s", annotation_id)
    ann = ls.annotations.get(id=annotation_id)
    if not ann:
        raise ExportCLIError(f"Annotation {annotation_id} not found")

    result = getattr(ann, "result", None)
    if not result:
        raise ReIDCLIError(f"Annotation {annotation_id} has no regions")

    logger.info(
        "Annotation fetched: id=%s with %d regions",
        getattr(ann, "id", annotation_id),
        len(result),
    )
    return ann


def _build_export_handle(ls_url: str, ls_api_key: str) -> ExportHandle:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Token {ls_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )
    base_url = ls_url.rstrip("/")
    return ExportHandle(base_url=base_url, session=session)


def _create_export_snapshot(handle: ExportHandle, project_id: int) -> int:
    url = f"{handle.base_url}/api/projects/{project_id}/exports"
    payload = {
        "title": "Export with Interpolated Keyframes",
        "serialization_options": {"interpolate_key_frames": True},
    }

    logger.info("Creating export snapshot with interpolated keyframes…")
    response = handle.session.post(url, data=json.dumps(payload), timeout=60)
    if response.status_code >= 400:
        raise ExportCLIError(
            f"Failed to create export snapshot (status={response.status_code}): {response.text}"
        )

    data = response.json()
    export_id = data.get("id") or data.get("pk")
    if export_id is None:
        raise ExportCLIError("Export creation response missing 'id'")

    logger.info("Export snapshot created: %s", export_id)
    return int(export_id)


def _poll_export_status(handle: ExportHandle, project_id: int, export_id: int, timeout_s: float, poll_interval: float) -> None:
    url = f"{handle.base_url}/api/projects/{project_id}/exports/{export_id}"
    logger.info("Waiting for export %s to finish…", export_id)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        response = handle.session.get(url, timeout=30)
        if response.status_code >= 400:
            raise ExportCLIError(
                f"Failed to poll export {export_id} (status={response.status_code}): {response.text}"
            )

        data = response.json()
        status = data.get("status") or data.get("state")
        logger.debug("Export %s status: %s", export_id, status)

        if status == "completed":
            logger.info("Export %s completed", export_id)
            return
        if status in {"failed", "error"}:
            raise ExportCLIError(f"Export {export_id} failed: {json.dumps(data, indent=2)}")

        time.sleep(max(0.5, poll_interval))

    raise ExportCLIError(
        f"Timed out after {timeout_s}s while waiting for export {export_id} to complete."
    )


def _download_export_json(handle: ExportHandle, project_id: int, export_id: int) -> List[Dict[str, Any]]:
    url = f"{handle.base_url}/api/projects/{project_id}/exports/{export_id}/download"
    params = {"exportType": "JSON"}
    logger.info("Downloading export %s as JSON…", export_id)
    response = handle.session.get(url, params=params, timeout=120)
    if response.status_code >= 400:
        raise ExportCLIError(
            f"Failed to download export {export_id} (status={response.status_code}): {response.text}"
        )

    try:
        return response.json()
    except json.JSONDecodeError:
        import io
        import zipfile

        logger.info("Export response is not JSON; attempting to read as ZIP archive…")
        bytes_buf = io.BytesIO(response.content)
        try:
            with zipfile.ZipFile(bytes_buf) as zf:
                first_name = zf.namelist()[0]
                with zf.open(first_name) as fp:
                    return json.load(fp)
        except zipfile.BadZipFile as exc:
            raise ExportCLIError(
                "Downloaded export is neither JSON nor a valid ZIP archive"
            ) from exc


def _normalize_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_annotation(
    export_payload: List[Dict[str, Any]],
    target_task_id: int,
    target_annotation_id: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for task in export_payload:
        task_id = _normalize_int(task.get("id") or task.get("task_id"))
        if task_id != target_task_id:
            continue

        annotations = task.get("annotations") or []
        for ann in annotations:
            ann_id = _normalize_int(ann.get("id"))
            if ann_id == target_annotation_id:
                filtered_task = {
                    "id": task.get("id"),
                    "data": task.get("data"),
                    "meta": task.get("meta"),
                    "annotations": [ann],
                }
                return ann, filtered_task

    raise ExportCLIError(
        f"Annotation {target_annotation_id} within task {target_task_id} not found in export payload"
    )


def _write_output(
    output_path: Path,
    project_id: int,
    task_id: int,
    annotation_id: int,
    annotation: Dict[str, Any],
    task_payload: Dict[str, Any],
):
    output_payload = {
        "project_id": project_id,
        "task_id": task_id,
        "annotation_id": annotation_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "annotation": annotation,
        "task": task_payload,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    logger.info("Saved filtered export to %s", output_path)


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("📤 EXPORT INTERPOLATED ANNOTATION CLI STARTED")
    logger.info("=" * 80)
    logger.info("📋 Parameters:")
    logger.info("   • Label Studio URL: %s", args.ls_url)
    logger.info("   • Project ID: %s", args.project)
    logger.info("   • Task ID: %s", args.task)
    logger.info("   • Annotation ID: %s", args.annotation)
    logger.info("=" * 80)

    exit_code = 0
    try:
        ls = _build_ls_client(args.ls_url, args.ls_api_key)
        _fetch_task(ls, args.project, args.task)
        _fetch_annotation(ls, args.annotation)

        handle = _build_export_handle(args.ls_url, args.ls_api_key)
        export_id = _create_export_snapshot(handle, args.project)
        _poll_export_status(handle, args.project, export_id, args.timeout, args.poll_interval)
        payload = _download_export_json(handle, args.project, export_id)
        annotation, task_payload = _extract_annotation(payload, args.task, args.annotation)

        if args.output is None:
            default_name = f"interpolated_project{args.project}_task{args.task}_ann{args.annotation}.json"
            output_path = Path.cwd() / default_name
        else:
            output_path = args.output

        _write_output(output_path, args.project, args.task, args.annotation, annotation, task_payload)

        logger.info("=" * 80)
        logger.info("✅ EXPORT INTERPOLATED ANNOTATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except ExportCLIError as e:
        logger.error("❌ Export CLI error: %s", e)
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
            logger.info("❌ EXPORT INTERPOLATED ANNOTATION FAILED (exit code: %s)", exit_code)
            logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
