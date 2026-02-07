"""Utilities for merging video regions in Label Studio predictions/annotations.

This module currently focuses on extracting numeric merge IDs from
Label Studio result objects. A merge ID is encoded in text fields as
strings like "id:31". Only well-formed numeric IDs are used.

The extraction logic is intentionally conservative:
- It looks primarily at result["meta"]["text"].
- It also considers result["text"] or result["value"]["text"] as
  reasonable fallbacks if present.
- It accepts variants such as "id:31", "id: 31", "ID:31".
- It ignores cases like "id:", "id: ", or any value where no digits
  follow the colon or where parsing to int fails.

This keeps the behavior aligned with the interactive JS plugin while
being robust to slightly different Label Studio payloads.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Configure root logger if not yet configured (for direct CLI use)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class MergeCLIError(Exception):
    """Custom error type for the merge CLI."""


# Compiled regex for patterns like "id:31", "id: 31", "ID:0031", etc.
_ID_PATTERN = re.compile(r"^\s*id\s*:\s*([0-9]+)\s*$", re.IGNORECASE)


def _extract_strings(field: Any) -> List[str]:
    """Normalize a potentially nested text field into a list of strings.

    Handles the following cases:
    - Single string -> [string]
    - List/tuple of strings -> [strings]
    - Anything else -> []
    """
    if isinstance(field, str):
        return [field]
    if isinstance(field, (list, tuple)):
        return [item for item in field if isinstance(item, str)]
    return []


def _ensure_meta_text_placeholder(result: Dict[str, Any]) -> None:
    """Ensure result["meta"]["text"] exists and has at least "id:".

    If meta/text is missing or contains only empty strings, this function
    initializes it to the placeholder string "id:" so the field is always
    present for downstream tooling.
    """
    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        result["meta"] = meta

    raw_text = meta.get("text")
    texts = _extract_strings(raw_text)

    # Treat completely missing text or only-empty strings as empty
    if not texts or all(not t.strip() for t in texts):
        meta["text"] = "id:"


def _normalize_text_candidates(result: Dict[str, Any]) -> List[str]:
    """Collect candidate text strings from a Label Studio result object.

    Priority order:
    1. result["meta"]["text"]
    2. result["text"] (if present)
    3. result["value"]["text"] (if present)

    All values are flattened into a list of strings, with duplicates
    removed while preserving order.
    """
    candidates: List[str] = []

    # Primary source: meta.text
    meta = result.get("meta")
    if isinstance(meta, dict):
        candidates.extend(_extract_strings(meta.get("text")))

    # Fallback: direct text field on result
    candidates.extend(_extract_strings(result.get("text")))

    # Fallback: value.text (common in LS results)
    value = result.get("value")
    if isinstance(value, dict):
        candidates.extend(_extract_strings(value.get("text")))

    # Deduplicate while preserving order
    seen = set()
    unique_candidates: List[str] = []
    for text in candidates:
        if text not in seen:
            seen.add(text)
            unique_candidates.append(text)

    return unique_candidates


def _parse_id_from_string(text: str) -> Optional[int]:
    """Parse a single string for an "id:<number>" pattern.

    Returns an integer if a valid numeric ID is found, otherwise None.

    Examples:
      "id:31"   -> 31
      "id: 31"  -> 31
      "ID:0031" -> 31
      "id:"     -> None
      "id: "    -> None
      "foo"     -> None
    """
    match = _ID_PATTERN.match(text)
    if not match:
        return None

    raw_number = match.group(1)
    try:
        return int(raw_number)
    except (TypeError, ValueError):
        logger.debug("Failed to parse numeric ID from %r", text)
        return None


def extract_merge_key_from_result(result: Dict[str, Any]) -> Optional[int]:
    """Extract the integer merge key from a Label Studio result object.

    The function searches for a textual marker of the form "id:<number>"
    in a small set of candidate locations on the result:

    - result["meta"]["text"] (primary)
    - result["text"] (fallback)
    - result["value"]["text"] (fallback)

    Only when a valid integer can be parsed from such a string is the
    value returned. All other cases (missing text, "id:" with no number,
    malformed content) result in None.

    Parameters
    ----------
    result:
        A single result/region dictionary from a Label Studio
        annotation or prediction.

    Returns
    -------
    Optional[int]
        The parsed numeric ID if one is found, or None otherwise.
    """
    result_id = result.get("id")

    # Ensure meta.text exists so the UI always has a placeholder field
    _ensure_meta_text_placeholder(result)

    for text in _normalize_text_candidates(result):
        merge_id = _parse_id_from_string(text)
        if merge_id is not None:
            logger.debug(
                "Extracted merge id %s from result %r via text %r",
                merge_id,
                result_id,
                text,
            )
            return merge_id

    logger.debug("No valid merge id found for result %r", result_id)
    return None


def extract_merge_keys_for_results(
    results: Iterable[Dict[str, Any]],
) -> Dict[str, int]:
    """Build a mapping from result/region IDs to merge IDs.

    This is a convenience wrapper around :func:`extract_merge_key_from_result`
    that operates on an iterable of results (such as annotation["result"]).

    Only results for which a valid merge ID is found are included.

    Parameters
    ----------
    results:
        Iterable of Label Studio result dictionaries.

    Returns
    -------
    Dict[str, int]
        Mapping from result["id"] (track ID) to parsed integer merge ID.
        Results without an "id" field or without a valid merge ID are
        skipped.
    """
    mapping: Dict[str, int] = {}

    for result in results:
        if not isinstance(result, dict):
            continue

        merge_id = extract_merge_key_from_result(result)
        if merge_id is None:
            continue

        track_id = result.get("id")
        if not isinstance(track_id, str):
            logger.debug("Result without string 'id' field skipped: %r", result)
            continue

        mapping[track_id] = merge_id

    return mapping


def _merge_sequences(sequences: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge, sort, and deduplicate keyframe sequences.

    Frames are ordered primarily by the "frame" field (if present) and
    secondarily by "time". Deduplication keeps the first occurrence of
    each frame (or time when frame is missing).
    """
    flat: List[Dict[str, Any]] = [s for s in sequences if isinstance(s, dict)]
    if not flat:
        return []

    def _sort_key(item: Dict[str, Any]):
        frame = item.get("frame")
        time_val = item.get("time")
        # Items without frame are sorted after items with frame
        return (frame is None, frame if isinstance(frame, int) else 0, float(time_val) if isinstance(time_val, (int, float)) else 0.0)

    flat.sort(key=_sort_key)

    seen = set()
    merged: List[Dict[str, Any]] = []
    for item in flat:
        frame = item.get("frame")
        time_val = item.get("time")
        if isinstance(frame, int):
            key = ("frame", frame)
        elif isinstance(time_val, (int, float)):
            key = ("time", float(time_val))
        else:
            # If neither frame nor time is present, keep as-is but don't dedup
            merged.append(item)
            continue

        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    return merged


def _merge_group(results: List[Dict[str, Any]], indices: List[int]) -> Dict[str, Any]:
    """Merge a group of results (same merge ID) into a single region.

    The first result in the group is used as a template; its structure
    (from_name, to_name, type, meta, etc.) is preserved while the
    sequence and labels are merged from all group members.
    """
    if not indices:
        raise ValueError("merge group indices must not be empty")

    base = deepcopy(results[indices[0]])
    base_value = base.setdefault("value", {})

    all_sequences: List[Dict[str, Any]] = []
    labels_union: List[str] = []
    labels_seen = set()
    frames_count_candidates: List[int] = []
    duration_candidates: List[float] = []

    for idx in indices:
        res = results[idx]
        value = res.get("value") or {}

        seq = value.get("sequence") or []
        if isinstance(seq, list):
            all_sequences.extend([s for s in seq if isinstance(s, dict)])

        labels = value.get("labels") or []
        if isinstance(labels, list):
            for label in labels:
                if not isinstance(label, str):
                    continue
                cleaned = label.strip()
                if cleaned and cleaned not in labels_seen:
                    labels_seen.add(cleaned)
                    labels_union.append(cleaned)

        frames_count = value.get("framesCount")
        if isinstance(frames_count, int):
            frames_count_candidates.append(frames_count)

        duration = value.get("duration")
        if isinstance(duration, (int, float)):
            duration_candidates.append(float(duration))

    if all_sequences:
        merged_sequence = _merge_sequences(all_sequences)
        base_value["sequence"] = merged_sequence

        # framesCount: prefer explicit framesCount from values, else derive from frames
        if not isinstance(base_value.get("framesCount"), int):
            if frames_count_candidates:
                base_value["framesCount"] = max(frames_count_candidates)
            else:
                frame_vals = [s.get("frame") for s in merged_sequence if isinstance(s.get("frame"), int)]
                if frame_vals:
                    base_value["framesCount"] = max(frame_vals)

        # duration: prefer existing durations, else derive from time values
        if not isinstance(base_value.get("duration"), (int, float)):
            if duration_candidates:
                base_value["duration"] = max(duration_candidates)
            else:
                time_vals = [s.get("time") for s in merged_sequence if isinstance(s.get("time"), (int, float))]
                if time_vals:
                    base_value["duration"] = float(max(time_vals))

    if labels_union:
        base_value["labels"] = labels_union

    logger.debug(
        "Merged group %s into result id=%r with %d keyframes",
        indices,
        base.get("id"),
        len(base_value.get("sequence", [])),
    )
    return base


def merge_results_by_merge_id(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge regions in a Label Studio result list by their numeric merge ID.

    The merge ID is extracted via :func:`extract_merge_key_from_result`.
    For each distinct merge ID with more than one region, all of that
    group's regions are replaced by a single merged region whose
    sequence and labels are the combination of the originals.

    Regions without a merge ID, or groups with only a single region,
    are preserved as-is.
    """
    if not results:
        logger.info("No results provided for merging; returning empty list")
        return []

    merge_ids: List[Optional[int]] = []
    groups: Dict[int, List[int]] = {}

    for idx, res in enumerate(results):
        if not isinstance(res, dict):
            merge_ids.append(None)
            continue

        merge_id = extract_merge_key_from_result(res)
        merge_ids.append(merge_id)
        if merge_id is None:
            continue
        groups.setdefault(merge_id, []).append(idx)

    if not groups:
        logger.info("No merge IDs found in results; returning original list")
        return list(results)

    merged_results: List[Dict[str, Any]] = []
    merged_group_count = 0

    for idx, res in enumerate(results):
        merge_id = merge_ids[idx]
        if merge_id is None:
            merged_results.append(res)
            continue

        indices = groups.get(merge_id) or []
        if len(indices) == 1:
            # Single region for this merge ID; keep as-is
            merged_results.append(res)
            continue

        # Only merge once per group, at the first index
        if idx != indices[0]:
            continue

        merged_group_count += 1
        merged_results.append(_merge_group(results, indices))

    logger.info(
        "Merge complete: total_regions_before=%d, total_regions_after=%d, merged_groups=%d",
        len(results),
        len(merged_results),
        merged_group_count,
    )
    return merged_results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the merge utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge Label Studio video regions that share the same numeric ID "
            "(e.g. 'id:31' in meta.text) into single tracks and upload a new prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python mergevideoregions.py --ls-url https://app.heartex.com \\\n"
            "    --ls-api-key YOUR_KEY --project 123 --task 456 --annotation 789\n\n"
            "  python mergevideoregions.py --ls-url https://app.heartex.com \\\n"
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
        help="Project ID (used for validation/logging)",
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
        help="Annotation ID to use as the source of regions",
    )
    source_group.add_argument(
        "--prediction",
        type=int,
        help="Prediction ID to use as the source of regions",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def _build_ls_client(ls_url: str, ls_api_key: str):
    """Create a Label Studio SDK client.

    Performs basic validation of the API key and sets a reasonable timeout.
    """
    if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
        raise MergeCLIError(
            "LABEL_STUDIO_API_KEY is required. "
            "Provide it via --ls-api-key or the LABEL_STUDIO_API_KEY env var."
        )

    # Lazy import to avoid hard dependency when used as a pure utility
    from label_studio_sdk.client import LabelStudio

    logger.info("Connecting to Label Studio at %s", ls_url)
    client = LabelStudio(base_url=ls_url, api_key=ls_api_key, timeout=600)
    logger.info("Connected to Label Studio")
    return client


def _fetch_task(ls, project_id: int, task_id: int) -> Dict[str, Any]:
    """Fetch and minimally validate a task from Label Studio."""
    logger.info("Fetching task %s from project %s", task_id, project_id)
    task_obj = ls.tasks.get(task_id)
    if not task_obj:
        raise MergeCLIError(f"Task {task_id} not found")

    task = {"id": task_obj.id, "data": getattr(task_obj, "data", {})}
    logger.info("Task fetched: %s", task.get("id"))
    return task


def _fetch_source_regions(ls, source_type: str, source_id: int) -> Dict[str, Any]:
    """Fetch annotation or prediction and return a dict with its regions.

    Returns a dict of the form {"id": <id>, "result": <result_list>, "kind": <str>}.
    """
    if source_type == "annotation":
        logger.info("Fetching annotation %s", source_id)
        obj = ls.annotations.get(id=source_id)
        kind = "Annotation"
    else:
        logger.info("Fetching prediction %s", source_id)
        obj = ls.predictions.get(id=source_id)
        kind = "Prediction"

    if not obj:
        raise MergeCLIError(f"{kind} {source_id} not found")

    result = getattr(obj, "result", None)
    if not result:
        raise MergeCLIError(f"{kind} {source_id} has no regions to merge")

    obj_id = getattr(obj, "id", source_id)
    logger.info("%s fetched: %s with %d regions", kind, obj_id, len(result))
    return {"id": obj_id, "result": result, "kind": kind}


def _build_prediction_data(
    merged_results: List[Dict[str, Any]], source_type: str, source_id: int
) -> Dict[str, Any]:
    """Build a prediction payload for the merged regions."""
    model_version = f"merged-{source_type}-{source_id}"
    prediction = {
        "result": merged_results,
        "score": 1.0,
        "model_version": model_version,
    }
    logger.debug(
        "Built prediction payload: model_version=%s, regions=%d",
        model_version,
        len(merged_results),
    )
    return prediction


def _upload_merged_prediction(ls, task_id: int, prediction_data: Dict[str, Any]):
    """Upload merged prediction to Label Studio using ls.predictions.create."""
    logger.info(
        "Uploading merged prediction for task %s (model_version=%s, regions=%d)...",
        task_id,
        prediction_data.get("model_version"),
        len(prediction_data.get("result", [])),
    )

    result = ls.predictions.create(
        task=task_id,
        score=prediction_data.get("score", 0),
        model_version=prediction_data.get("model_version", "merged"),
        result=prediction_data.get("result", []),
    )

    pred_id = getattr(result, "id", None)
    if pred_id is not None:
        logger.info("Upload complete, prediction id=%s", pred_id)
    else:
        logger.info("Upload request completed (no prediction id in response)")

    return result


def main() -> None:
    """CLI entrypoint to merge video regions and create a new prediction."""
    args = _parse_args()

    # Set global log level according to CLI argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("🚀 VIDEO REGION MERGE CLI STARTED")
    logger.info("=" * 80)
    logger.info("📋 Parameters:")
    logger.info("   • Label Studio URL: %s", args.ls_url)
    logger.info("   • Project ID: %s", args.project)
    logger.info("   • Task ID: %s", args.task)
    if args.annotation is not None:
        logger.info("   • Source: annotation %s", args.annotation)
    if args.prediction is not None:
        logger.info("   • Source: prediction %s", args.prediction)
    logger.info("=" * 80)

    exit_code = 0

    try:
        ls = _build_ls_client(args.ls_url, args.ls_api_key)

        # Basic validation that the task exists (project ID is used only for logging)
        _fetch_task(ls, args.project, args.task)

        if args.annotation is not None:
            source_type = "annotation"
            source_id = args.annotation
        else:
            source_type = "prediction"
            source_id = args.prediction  # type: ignore[assignment]

        source = _fetch_source_regions(ls, source_type, source_id)

        original_count = len(source["result"])
        logger.info("Merging regions by numeric ID (original count=%d)", original_count)

        merged_results = merge_results_by_merge_id(source["result"])
        merged_count = len(merged_results)

        logger.info(
            "Merge summary: original_regions=%d, merged_regions=%d",
            original_count,
            merged_count,
        )

        prediction_data = _build_prediction_data(merged_results, source_type, source_id)
        _upload_merged_prediction(ls, args.task, prediction_data)

        logger.info("=" * 80)
        logger.info("✅ MERGE CLI EXECUTION SUCCESSFUL")
        logger.info("=" * 80)

    except MergeCLIError as e:
        logger.error("❌ Merge CLI error: %s", e)
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
            logger.info("❌ MERGE CLI EXECUTION FAILED (exit code: %s)", exit_code)
            logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
