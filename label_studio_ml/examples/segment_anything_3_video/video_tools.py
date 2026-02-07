from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class VideoToolsError(Exception):
    pass


@dataclass(frozen=True)
class LabelStudioAPI:
    base_url: str
    session: requests.Session


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return None


def _build_ls_api(ls_url: Optional[str], ls_api_key: Optional[str]) -> LabelStudioAPI:
    resolved_url = (ls_url or _env_first("LABEL_STUDIO_URL", "LABEL_STUDIO_HOST"))
    resolved_key = (ls_api_key or _env_first("LABEL_STUDIO_API_KEY"))

    if not resolved_url or resolved_url.strip() == "":
        raise VideoToolsError(
            "Label Studio URL is required. Provide --ls-url or set LABEL_STUDIO_URL (or LABEL_STUDIO_HOST)."
        )
    if not resolved_key or resolved_key.strip() == "":
        raise VideoToolsError(
            "Label Studio API key is required. Provide --ls-api-key or set LABEL_STUDIO_API_KEY."
        )

    base_url = resolved_url.rstrip("/")
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Token {resolved_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
    )
    return LabelStudioAPI(base_url=base_url, session=session)


def _read_json_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return None


def _fetch_annotation(api: LabelStudioAPI, task_id: int, annotation_id: int) -> Dict[str, Any]:
    task_url = f"{api.base_url}/api/tasks/{task_id}/annotations/{annotation_id}/"
    response = api.session.get(task_url, timeout=60)

    if response.status_code == 404:
        ann_url = f"{api.base_url}/api/annotations/{annotation_id}/"
        response = api.session.get(ann_url, timeout=60)

    if response.status_code >= 400:
        raise VideoToolsError(
            f"Failed to fetch annotation (status={response.status_code}). URL={response.url}"
        )

    data = _read_json_response(response)
    if not isinstance(data, dict):
        raise VideoToolsError("Annotation response is not a JSON object")

    result = data.get("result")
    if result is None:
        raise VideoToolsError(f"Annotation {annotation_id} JSON missing 'result'")
    if not isinstance(result, list):
        raise VideoToolsError(f"Annotation {annotation_id} 'result' is not a list")

    return data


def _patch_annotation(
    api: LabelStudioAPI,
    annotation_id: int,
    result: Sequence[Dict[str, Any]],
) -> None:
    url = f"{api.base_url}/api/annotations/{annotation_id}/"
    payload = {"result": list(result)}

    try:
        response = api.session.patch(url, data=json.dumps(payload), timeout=180)
    except requests.exceptions.RequestException as exc:
        raise VideoToolsError(f"PATCH {url} failed: {exc}") from exc

    if response.status_code in {200, 201, 202, 204}:
        logger.info("Annotation updated successfully (annotation=%s)", annotation_id)
        return

    if response.status_code == 504:
        logger.warning(
            "Received 504 Gateway Timeout from Label Studio. "
            "Data was sent successfully; treating as success (annotation=%s).",
            annotation_id,
        )
        return

    raise VideoToolsError(
        f"PATCH {url} failed (status={response.status_code}): {response.text[:300]}"
    )


def _get_sequence(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = result.get("value")
    if not isinstance(value, dict):
        return []
    seq = value.get("sequence")
    if not isinstance(seq, list):
        return []
    return [item for item in seq if isinstance(item, dict)]


def _set_sequence(result: Dict[str, Any], sequence: Sequence[Dict[str, Any]]) -> None:
    value = result.get("value")
    if not isinstance(value, dict):
        value = {}
        result["value"] = value
    value["sequence"] = list(sequence)


def _get_frame(item: Dict[str, Any]) -> Optional[int]:
    try:
        frame = int(item.get("frame"))
    except (TypeError, ValueError):
        return None
    if frame <= 0:
        return None
    return frame


def _find_unique_track(results: List[Dict[str, Any]], track_id: str) -> Dict[str, Any]:
    track_id = str(track_id)
    if track_id.strip() == "":
        raise VideoToolsError("Track id must be a non-empty string")

    matches: List[Dict[str, Any]] = []
    for res in results:
        if not isinstance(res, dict):
            continue
        if res.get("id") == track_id:
            matches.append(res)

    if not matches:
        available = [
            res.get("id")
            for res in results
            if isinstance(res, dict) and isinstance(res.get("id"), str)
        ]
        sample = ", ".join(available[:20])
        suffix = "" if len(available) <= 20 else f", ... (+{len(available) - 20} more)"
        hint = f" Available track ids include: {sample}{suffix}" if available else ""
        raise VideoToolsError(f"No track found with track id '{track_id}'.{hint}")
    if len(matches) > 1:
        raise VideoToolsError(
            f"Found {len(matches)} tracks with the same track id '{track_id}'. "
            "This should not happen; please inspect the annotation JSON."
        )
    return matches[0]


def _validate_frame_range(start_frame: int, end_frame: int) -> Tuple[int, int]:
    start = int(start_frame)
    end = int(end_frame)
    if start <= 0 or end <= 0:
        raise VideoToolsError("Frame indices must be positive (Label Studio frames are 1-based)")
    if start > end:
        raise VideoToolsError(f"start-frame ({start}) must be <= end-frame ({end})")
    return start, end


def _uniform_keep_indices(n: int, keep_count: int) -> List[int]:
    if n <= 0:
        return []
    keep_count = max(1, min(int(keep_count), n))
    if keep_count == 1:
        return [0]

    step = float(n - 1) / float(keep_count - 1)
    indices: List[int] = []
    for i in range(keep_count):
        idx = int(round(i * step))
        if indices and idx <= indices[-1]:
            idx = indices[-1] + 1
        if idx > n - 1:
            idx = n - 1
        indices.append(idx)
    return indices


def _sort_sequence(sequence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(item: Dict[str, Any]) -> Tuple[int, int]:
        frame = _get_frame(item)
        if frame is None:
            return (1, 0)
        return (0, frame)

    return sorted(list(sequence), key=_key)


def _update_track_stats(result: Dict[str, Any]) -> None:
    value = result.get("value")
    if not isinstance(value, dict):
        return

    sequence = _get_sequence(result)
    frames = [f for f in (_get_frame(item) for item in sequence) if f is not None]
    if frames:
        value["framesCount"] = max(frames)

    times: List[float] = []
    for item in sequence:
        time_val = item.get("time")
        if isinstance(time_val, (int, float)):
            times.append(float(time_val))
    if times:
        value["duration"] = max(times)


def _sparsify_sequence(
    sequence: List[Dict[str, Any]],
    start_frame: int,
    end_frame: int,
    ratio: float,
) -> Tuple[List[Dict[str, Any]], int, int]:
    if ratio <= 0 or ratio > 1:
        raise VideoToolsError("--ratio must be in (0, 1]")

    in_range: List[Tuple[int, Dict[str, Any]]] = []
    for item in sequence:
        frame = _get_frame(item)
        if frame is None:
            continue
        if start_frame <= frame <= end_frame:
            in_range.append((frame, item))

    in_range.sort(key=lambda pair: pair[0])
    total_in_range = len(in_range)
    if total_in_range == 0:
        return list(sequence), 0, 0

    keep_count = int(round(float(total_in_range) * float(ratio)))
    keep_count = max(1, min(keep_count, total_in_range))

    keep_indices = _uniform_keep_indices(total_in_range, keep_count)
    keep_frames = {in_range[idx][0] for idx in keep_indices}

    new_sequence: List[Dict[str, Any]] = []
    removed = 0
    for item in sequence:
        frame = _get_frame(item)
        if frame is None or frame < start_frame or frame > end_frame:
            new_sequence.append(item)
            continue
        if frame in keep_frames:
            new_sequence.append(item)
        else:
            removed += 1

    return new_sequence, removed, keep_count


def _clamp_box(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    w = max(0.0, min(100.0, w))
    h = max(0.0, min(100.0, h))

    if w >= 100.0:
        x = 0.0
    if h >= 100.0:
        y = 0.0

    x = max(0.0, min(100.0 - w, x))
    y = max(0.0, min(100.0 - h, y))

    return x, y, w, h


def _inflate_box(x: float, y: float, w: float, h: float, percent: float) -> Tuple[float, float, float, float]:
    scale = 1.0 + float(percent)
    if scale <= 0:
        raise VideoToolsError("--percent must be > -1.0")

    cx = x + 0.5 * w
    cy = y + 0.5 * h

    new_w = w * scale
    new_h = h * scale
    new_x = cx - 0.5 * new_w
    new_y = cy - 0.5 * new_h

    return _clamp_box(new_x, new_y, new_w, new_h)


def _smooth_sequence(sequence: List[Dict[str, Any]], window: int) -> int:
    window = int(window)
    if window <= 0:
        raise VideoToolsError("--window must be > 0")

    indexed: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, item in enumerate(sequence):
        frame = _get_frame(item)
        if frame is None:
            continue
        if not bool(item.get("enabled", True)):
            continue
        indexed.append((idx, frame, item))

    indexed.sort(key=lambda t: t[1])
    if not indexed:
        return 0

    values: List[Tuple[float, float, float, float]] = []
    for _idx, _frame, item in indexed:
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
            w = float(item.get("width"))
            h = float(item.get("height"))
        except (TypeError, ValueError):
            values.append((float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        values.append((x, y, w, h))

    updated = 0
    n = len(indexed)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, start + window)
        start = max(0, end - window)

        xs: List[float] = []
        ys: List[float] = []
        ws: List[float] = []
        hs: List[float] = []
        for j in range(start, end):
            x, y, w, h = values[j]
            if any(val != val for val in (x, y, w, h)):
                continue
            xs.append(x)
            ys.append(y)
            ws.append(w)
            hs.append(h)

        if not xs:
            continue

        mean_x = sum(xs) / float(len(xs))
        mean_y = sum(ys) / float(len(ys))
        mean_w = sum(ws) / float(len(ws))
        mean_h = sum(hs) / float(len(hs))
        mean_x, mean_y, mean_w, mean_h = _clamp_box(mean_x, mean_y, mean_w, mean_h)

        seq_idx, _frame, item = indexed[i]
        item["x"] = mean_x
        item["y"] = mean_y
        item["width"] = mean_w
        item["height"] = mean_h
        sequence[seq_idx] = item
        updated += 1

    return updated


def _write_dry_run_output(
    annotation: Dict[str, Any],
    task_id: int,
    annotation_id: int,
    command: str,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"video_tools_{command}_task{task_id}_ann{annotation_id}_{ts}.json"
    path = Path.cwd() / filename
    path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    return path


def _apply_and_commit(
    api: LabelStudioAPI,
    annotation: Dict[str, Any],
    task_id: int,
    annotation_id: int,
    command: str,
    dry_run: bool,
) -> None:
    results = annotation.get("result")
    if not isinstance(results, list):
        raise VideoToolsError("Annotation JSON missing 'result' list")

    if dry_run:
        path = _write_dry_run_output(annotation, task_id, annotation_id, command)
        logger.info("Dry-run enabled. Wrote updated annotation JSON to %s", path)
        return

    _patch_annotation(api, annotation_id=annotation_id, result=results)


def _cmd_sparsify(api: LabelStudioAPI, args: argparse.Namespace) -> None:
    start_frame, end_frame = _validate_frame_range(args.start_frame, args.end_frame)

    annotation = _fetch_annotation(api, task_id=args.task, annotation_id=args.annotation)
    results = annotation["result"]

    track = _find_unique_track(results, args.track_id)
    sequence = _get_sequence(track)

    new_sequence, removed, kept = _sparsify_sequence(sequence, start_frame, end_frame, args.ratio)
    _set_sequence(track, _sort_sequence(new_sequence))
    _update_track_stats(track)

    logger.info(
        "Sparsify complete (track_id=%s, range=%s-%s, ratio=%.4f): kept=%d removed=%d",
        args.track_id,
        start_frame,
        end_frame,
        float(args.ratio),
        kept,
        removed,
    )

    _apply_and_commit(
        api,
        annotation=annotation,
        task_id=args.task,
        annotation_id=args.annotation,
        command="sparsify",
        dry_run=bool(args.dry_run),
    )


def _cmd_swap_ids(api: LabelStudioAPI, args: argparse.Namespace) -> None:
    start_frame, end_frame = _validate_frame_range(args.start_frame, args.end_frame)

    annotation = _fetch_annotation(api, task_id=args.task, annotation_id=args.annotation)
    results = annotation["result"]

    source = _find_unique_track(results, args.source_track_id)
    target = _find_unique_track(results, args.target_track_id)

    source_seq = _get_sequence(source)
    target_seq = _get_sequence(target)

    moved: List[Dict[str, Any]] = []
    remaining_source: List[Dict[str, Any]] = []
    for item in source_seq:
        frame = _get_frame(item)
        if frame is None or frame < start_frame or frame > end_frame:
            remaining_source.append(item)
        else:
            moved.append(item)

    if not moved:
        logger.info(
            "No frames to move (source_track_id=%s, range=%s-%s). No changes applied.",
            args.source_track_id,
            start_frame,
            end_frame,
        )
        return

    cleaned_target: List[Dict[str, Any]] = []
    for item in target_seq:
        frame = _get_frame(item)
        if frame is None or frame < start_frame or frame > end_frame:
            cleaned_target.append(item)

    merged_target = cleaned_target + moved

    _set_sequence(source, _sort_sequence(remaining_source))
    _update_track_stats(source)

    _set_sequence(target, _sort_sequence(merged_target))
    _update_track_stats(target)

    logger.info(
        "Swap-ids complete: moved=%d frame(s) from track_id=%s -> track_id=%s (range=%s-%s)",
        len(moved),
        args.source_track_id,
        args.target_track_id,
        start_frame,
        end_frame,
    )

    _apply_and_commit(
        api,
        annotation=annotation,
        task_id=args.task,
        annotation_id=args.annotation,
        command="swap_ids",
        dry_run=bool(args.dry_run),
    )


def _cmd_trim_tail(api: LabelStudioAPI, args: argparse.Namespace) -> None:
    cutoff = int(args.cutoff_frame)
    if cutoff <= 0:
        raise VideoToolsError("--cutoff-frame must be positive (Label Studio frames are 1-based)")

    annotation = _fetch_annotation(api, task_id=args.task, annotation_id=args.annotation)
    results = annotation["result"]

    track = _find_unique_track(results, args.track_id)
    sequence = _get_sequence(track)

    new_sequence: List[Dict[str, Any]] = []
    removed = 0
    for item in sequence:
        frame = _get_frame(item)
        if frame is not None and frame > cutoff:
            removed += 1
            continue
        new_sequence.append(item)

    _set_sequence(track, _sort_sequence(new_sequence))
    _update_track_stats(track)

    logger.info("Trim-tail complete (track_id=%s, cutoff=%s): removed=%d", args.track_id, cutoff, removed)

    _apply_and_commit(
        api,
        annotation=annotation,
        task_id=args.task,
        annotation_id=args.annotation,
        command="trim_tail",
        dry_run=bool(args.dry_run),
    )


def _cmd_smooth(api: LabelStudioAPI, args: argparse.Namespace) -> None:
    window = int(args.window)

    annotation = _fetch_annotation(api, task_id=args.task, annotation_id=args.annotation)
    results = annotation["result"]

    track = _find_unique_track(results, args.track_id)
    sequence = _get_sequence(track)

    updated = _smooth_sequence(sequence, window)
    _set_sequence(track, _sort_sequence(sequence))
    _update_track_stats(track)

    logger.info("Smooth complete (track_id=%s, window=%s): updated=%d", args.track_id, window, updated)

    _apply_and_commit(
        api,
        annotation=annotation,
        task_id=args.task,
        annotation_id=args.annotation,
        command="smooth",
        dry_run=bool(args.dry_run),
    )


def _cmd_pad(api: LabelStudioAPI, args: argparse.Namespace) -> None:
    start_frame, end_frame = _validate_frame_range(args.start_frame, args.end_frame)
    percent = float(args.percent)

    annotation = _fetch_annotation(api, task_id=args.task, annotation_id=args.annotation)
    results = annotation["result"]

    track = _find_unique_track(results, args.track_id)
    sequence = _get_sequence(track)

    updated = 0
    for item in sequence:
        frame = _get_frame(item)
        if frame is None or frame < start_frame or frame > end_frame:
            continue
        if not bool(item.get("enabled", True)):
            continue

        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
            w = float(item.get("width"))
            h = float(item.get("height"))
        except (TypeError, ValueError):
            continue

        x, y, w, h = _inflate_box(x, y, w, h, percent)
        item["x"] = x
        item["y"] = y
        item["width"] = w
        item["height"] = h
        updated += 1

    _set_sequence(track, _sort_sequence(sequence))
    _update_track_stats(track)

    logger.info(
        "Pad complete (track_id=%s, percent=%.4f, range=%s-%s): updated=%d",
        args.track_id,
        percent,
        start_frame,
        end_frame,
        updated,
    )

    _apply_and_commit(
        api,
        annotation=annotation,
        task_id=args.task,
        annotation_id=args.annotation,
        command="pad",
        dry_run=bool(args.dry_run),
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ls-url",
        default=None,
        help="Label Studio URL (or env LABEL_STUDIO_URL / LABEL_STUDIO_HOST)",
    )
    parser.add_argument(
        "--ls-api-key",
        default=None,
        help="Label Studio API key (or env LABEL_STUDIO_API_KEY)",
    )
    parser.add_argument("--task", type=int, required=True, help="Task ID")
    parser.add_argument("--annotation", type=int, required=True, help="Annotation ID")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, write updated annotation JSON to a local file instead of PATCHing Label Studio",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common_args(common)

    parser = argparse.ArgumentParser(
        description="Label Studio video annotation toolbox (operates on videorectangle track sequences)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sparsify = subparsers.add_parser(
        "sparsify",
        parents=[common],
        help="Uniformly downsample frames in a range for one track",
    )
    sparsify.add_argument(
        "--track-id",
        type=str,
        required=True,
        help="Region track id from the annotation result list (e.g., auto-track-0)",
    )
    sparsify.add_argument("--start-frame", type=int, required=True)
    sparsify.add_argument("--end-frame", type=int, required=True)
    sparsify.add_argument("--ratio", type=float, required=True, help="Fraction of frames to keep (0,1]")

    swap_ids = subparsers.add_parser(
        "swap-ids",
        parents=[common],
        help="Move a segment of track history from one track id to another",
    )
    swap_ids.add_argument("--source-track-id", type=str, required=True)
    swap_ids.add_argument("--target-track-id", type=str, required=True)
    swap_ids.add_argument("--start-frame", type=int, required=True)
    swap_ids.add_argument("--end-frame", type=int, required=True)

    trim_tail = subparsers.add_parser(
        "trim-tail",
        parents=[common],
        help="Delete all frames after a cutoff for one track",
    )
    trim_tail.add_argument("--track-id", type=str, required=True)
    trim_tail.add_argument("--cutoff-frame", type=int, required=True)

    smooth = subparsers.add_parser(
        "smooth",
        parents=[common],
        help="Smooth x/y/width/height using a moving average window",
    )
    smooth.add_argument("--track-id", type=str, required=True)
    smooth.add_argument("--window", type=int, default=5)

    pad = subparsers.add_parser(
        "pad",
        parents=[common],
        help="Inflate boxes by a percentage over a frame range",
    )
    pad.add_argument("--track-id", type=str, required=True)
    pad.add_argument("--percent", type=float, required=True, help="e.g., 0.10 expands by 10%")
    pad.add_argument("--start-frame", type=int, required=True)
    pad.add_argument("--end-frame", type=int, required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    api = _build_ls_api(ls_url=args.ls_url, ls_api_key=args.ls_api_key)

    if args.command == "sparsify":
        _cmd_sparsify(api, args)
        return

    if args.command == "swap-ids":
        _cmd_swap_ids(api, args)
        return

    if args.command == "trim-tail":
        _cmd_trim_tail(api, args)
        return

    if args.command == "smooth":
        _cmd_smooth(api, args)
        return

    if args.command == "pad":
        _cmd_pad(api, args)
        return

    raise VideoToolsError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except VideoToolsError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(130)
