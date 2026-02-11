"""Annotation write-back: apply identity results to Label Studio annotations.

Handles 5 ID-switching scenarios:
1. ID swap (crossover)
2. Fragment merge (non-overlapping same-identity tracks)
3. Track split (mixed-identity track)
4. Keyframe move (outlier keyframe to correct track)
5. Collision (replace existing keyframe at same frame)

Plus: ID labeling (meta.text = "id:<number>")
"""

from __future__ import annotations

import copy
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ===========================================================================
# Identity assignment from per-crop identities
# ===========================================================================

def compute_track_identities(
    per_crop_identity: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-track identity from per-crop identity via majority vote.

    Args:
        per_crop_identity: {crop_id -> identity}. Crop IDs have format
            "{region_id}_f{frame}".

    Returns:
        {track_region_id: {majority_identity, is_mixed, identity_counts, per_frame}}
    """
    # Group by track
    track_crops: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for crop_id, identity in per_crop_identity.items():
        parts = crop_id.rsplit("_f", 1)
        if len(parts) != 2:
            continue
        region_id = parts[0]
        frame = int(parts[1])
        track_crops[region_id].append((frame, identity))

    result = {}
    for region_id, frame_identities in track_crops.items():
        frame_identities.sort(key=lambda x: x[0])
        counts = Counter(ident for _, ident in frame_identities)
        majority_identity = counts.most_common(1)[0][0]
        total = sum(counts.values())
        majority_count = counts[majority_identity]
        # Mixed if majority is less than 70% of total
        is_mixed = (majority_count / total) < 0.70 if total > 0 else False

        per_frame = {frame: ident for frame, ident in frame_identities}
        result[region_id] = {
            "majority_identity": majority_identity,
            "is_mixed": is_mixed,
            "identity_counts": dict(counts),
            "per_frame": per_frame,
        }

    return result


# ===========================================================================
# Scenario 1: ID Swap detection
# ===========================================================================

def detect_id_swaps(
    per_crop_identity: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Detect tracks that swap identities at a crossover point.

    Two tracks swap if:
    - Track A has identity X before frame F and identity Y after
    - Track B has identity Y before frame F and identity X after
    """
    track_info = compute_track_identities(per_crop_identity)

    # Find tracks with identity transitions
    transitioning = []
    for region_id, info in track_info.items():
        if not info["is_mixed"] and len(info["identity_counts"]) <= 1:
            continue
        per_frame = info["per_frame"]
        frames = sorted(per_frame.keys())
        if len(frames) < 2:
            continue

        # Find transition points
        for i in range(1, len(frames)):
            if per_frame[frames[i]] != per_frame[frames[i - 1]]:
                transitioning.append({
                    "region_id": region_id,
                    "crossover_frame": frames[i],
                    "before_identity": per_frame[frames[i - 1]],
                    "after_identity": per_frame[frames[i]],
                })

    # Match complementary transitions
    swaps = []
    used = set()
    for i, t1 in enumerate(transitioning):
        if i in used:
            continue
        for j, t2 in enumerate(transitioning):
            if j in used or j <= i:
                continue
            if t1["region_id"] == t2["region_id"]:
                continue
            # Check complementary: A goes X->Y, B goes Y->X at similar frame
            if (t1["before_identity"] == t2["after_identity"] and
                    t1["after_identity"] == t2["before_identity"] and
                    abs(t1["crossover_frame"] - t2["crossover_frame"]) <= 5):
                swaps.append({
                    "track_a": t1["region_id"],
                    "track_b": t2["region_id"],
                    "crossover_frame": min(t1["crossover_frame"], t2["crossover_frame"]),
                })
                used.add(i)
                used.add(j)
                break

    return swaps


def apply_id_swap(
    result: List[Dict[str, Any]],
    swap: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Swap keyframes between two tracks from crossover_frame onward."""
    result = copy.deepcopy(result)
    track_a_region = None
    track_b_region = None

    for region in result:
        if region["id"] == swap["track_a"]:
            track_a_region = region
        elif region["id"] == swap["track_b"]:
            track_b_region = region

    if not track_a_region or not track_b_region:
        return result

    crossover = swap["crossover_frame"]
    seq_a = track_a_region["value"]["sequence"]
    seq_b = track_b_region["value"]["sequence"]

    # Separate keyframes before/after crossover
    a_before = [k for k in seq_a if k["frame"] < crossover]
    a_after = [k for k in seq_a if k["frame"] >= crossover]
    b_before = [k for k in seq_b if k["frame"] < crossover]
    b_after = [k for k in seq_b if k["frame"] >= crossover]

    # Swap: A keeps before + B's after; B keeps before + A's after
    track_a_region["value"]["sequence"] = sorted(
        a_before + b_after, key=lambda k: k["frame"]
    )
    track_b_region["value"]["sequence"] = sorted(
        b_before + a_after, key=lambda k: k["frame"]
    )

    return result


# ===========================================================================
# Scenario 2: Fragment Merge
# ===========================================================================

def detect_fragment_merges(
    track_info: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect non-overlapping tracks with the same identity for merging.

    Args:
        track_info: {region_id: {frames: [frame_nums], identity: int}}
    """
    # Group by identity
    identity_tracks: Dict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
    for region_id, info in track_info.items():
        identity_tracks[info["identity"]].append((region_id, info["frames"]))

    merges = []
    for identity, tracks in identity_tracks.items():
        if len(tracks) < 2:
            continue

        # Sort by earliest frame
        tracks.sort(key=lambda t: min(t[1]))

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                rid_a, frames_a = tracks[i]
                rid_b, frames_b = tracks[j]
                # Check non-overlapping
                overlap = set(frames_a) & set(frames_b)
                if len(overlap) == 0:
                    merges.append({
                        "keep_track": rid_a,
                        "merge_track": rid_b,
                    })

    return merges


def apply_fragment_merge(
    result: List[Dict[str, Any]],
    merge: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Merge keyframes from merge_track into keep_track, remove merge_track."""
    result = copy.deepcopy(result)
    keep_region = None
    merge_region = None

    for region in result:
        if region["id"] == merge["keep_track"]:
            keep_region = region
        elif region["id"] == merge["merge_track"]:
            merge_region = region

    if not keep_region or not merge_region:
        return result

    # Add merge_track's keyframes to keep_track
    combined = keep_region["value"]["sequence"] + merge_region["value"]["sequence"]
    combined.sort(key=lambda k: k["frame"])
    keep_region["value"]["sequence"] = combined

    # Remove merge_track
    result = [r for r in result if r["id"] != merge["merge_track"]]

    return result


# ===========================================================================
# Scenario 3: Track Split
# ===========================================================================

def detect_track_splits(
    per_crop_identity: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Detect tracks with mixed identities that need splitting."""
    track_info = compute_track_identities(per_crop_identity)

    splits = []
    for region_id, info in track_info.items():
        if len(info["identity_counts"]) < 2:
            continue

        per_frame = info["per_frame"]
        frames = sorted(per_frame.keys())

        # Build temporally coherent segments
        segments: List[Dict[str, Any]] = []
        current_identity = per_frame[frames[0]]
        current_frames = [frames[0]]

        for i in range(1, len(frames)):
            if per_frame[frames[i]] == current_identity:
                current_frames.append(frames[i])
            else:
                segments.append({
                    "identity": current_identity,
                    "frames": list(current_frames),
                })
                current_identity = per_frame[frames[i]]
                current_frames = [frames[i]]
        segments.append({
            "identity": current_identity,
            "frames": list(current_frames),
        })

        if len(segments) >= 2:
            # Determine which identity segment to keep on the original track
            # Keep the segment with the most frames
            keep_identity = max(
                info["identity_counts"].items(),
                key=lambda x: x[1],
            )[0]
            splits.append({
                "source_track": region_id,
                "keep_identity": keep_identity,
                "segments": segments,
            })

    return splits


def apply_track_split(
    result: List[Dict[str, Any]],
    split: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Split a track into separate regions, one per identity segment."""
    result = copy.deepcopy(result)
    source = None
    source_idx = None
    for i, region in enumerate(result):
        if region["id"] == split["source_track"]:
            source = region
            source_idx = i
            break

    if source is None:
        return result

    seq = source["value"]["sequence"]
    segments = split["segments"]

    # Build frame lookup
    frame_to_kf = {k["frame"]: k for k in seq}

    # First segment stays on original track
    first_segment_frames = set(segments[0]["frames"])
    source["value"]["sequence"] = sorted(
        [k for k in seq if k["frame"] in first_segment_frames],
        key=lambda k: k["frame"],
    )

    # Subsequent segments become new tracks
    import uuid
    for seg in segments[1:]:
        seg_frames = set(seg["frames"])
        new_kfs = sorted(
            [k for k in seq if k["frame"] in seg_frames],
            key=lambda k: k["frame"],
        )
        if not new_kfs:
            continue

        new_region = copy.deepcopy(source)
        new_region["id"] = f"reid-split-{uuid.uuid4().hex[:8]}"
        new_region["value"]["sequence"] = new_kfs
        result.append(new_region)

    return result


# ===========================================================================
# Scenario 4 + 5: Keyframe Move (with collision handling)
# ===========================================================================

def apply_keyframe_move(
    result: List[Dict[str, Any]],
    move: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Move a keyframe from source_track to target_track at the given frame.

    If target_track already has a keyframe at that frame (collision),
    replace it with the moved keyframe.
    """
    result = copy.deepcopy(result)
    source = None
    target = None

    for region in result:
        if region["id"] == move["source_track"]:
            source = region
        elif region["id"] == move["target_track"]:
            target = region

    if not source or not target:
        return result

    frame = move["frame"]

    # Find the keyframe to move
    moved_kf = None
    new_source_seq = []
    for kf in source["value"]["sequence"]:
        if kf["frame"] == frame and moved_kf is None:
            moved_kf = kf
        else:
            new_source_seq.append(kf)
    source["value"]["sequence"] = new_source_seq

    if moved_kf is None:
        return result

    # Remove existing keyframe at this frame from target (collision handling)
    target["value"]["sequence"] = [
        kf for kf in target["value"]["sequence"] if kf["frame"] != frame
    ]

    # Add the moved keyframe
    target["value"]["sequence"].append(moved_kf)
    target["value"]["sequence"].sort(key=lambda k: k["frame"])

    return result


# ===========================================================================
# ID Labeling
# ===========================================================================

def apply_id_labels(
    result: List[Dict[str, Any]],
    identity_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Set meta.text to include "id:<number>" for each region.

    If the meta text already has "id:" followed by content, update it.
    If "id:" is present but empty, append the number.
    If no "id:" prefix exists, add it.
    """
    result = copy.deepcopy(result)

    for region in result:
        region_id = region["id"]
        if region_id not in identity_map:
            continue

        identity = identity_map[region_id]

        # Ensure meta dict exists
        if not isinstance(region.get("meta"), dict):
            region["meta"] = {}
        meta = region["meta"]

        raw_text = meta.get("text", "")
        if isinstance(raw_text, list):
            raw_text = " ".join(str(t) for t in raw_text)
        raw_text = str(raw_text) if raw_text else ""

        # Update or add id:<number>
        id_pattern = re.compile(r"id:\S*")
        if id_pattern.search(raw_text):
            # Replace existing id: value
            new_text = id_pattern.sub(f"id:{identity}", raw_text)
        elif "id:" in raw_text:
            # "id:" exists but empty — append number
            new_text = raw_text.replace("id:", f"id:{identity}", 1)
        else:
            # No id: prefix at all — add it
            new_text = f"{raw_text} id:{identity}".strip()

        meta["text"] = new_text

    return result


# ===========================================================================
# Full write-back orchestration
# ===========================================================================

def compute_writeback_preview(
    annotation_result: List[Dict[str, Any]],
    per_crop_identity: Dict[str, int],
    img_w: int,
    img_h: int,
) -> Dict[str, Any]:
    """Compute a dry-run preview of all annotation mutations.

    Returns a summary of planned changes without modifying anything.
    """
    track_info = compute_track_identities(per_crop_identity)

    mutations: List[Dict[str, Any]] = []

    # Build simplified track info for fragment merge detection
    simple_track_info: Dict[str, Dict[str, Any]] = {}
    for region in annotation_result:
        if region.get("type") != "videorectangle":
            continue
        rid = region["id"]
        frames = [int(k["frame"]) for k in region["value"].get("sequence", [])]
        if rid in track_info:
            simple_track_info[rid] = {
                "frames": frames,
                "identity": track_info[rid]["majority_identity"],
            }

    # Detect scenarios
    splits = detect_track_splits(per_crop_identity)
    for s in splits:
        mutations.append({
            "type": "track_split",
            "description": f"Split {s['source_track']} into {len(s['segments'])} segments",
            "source_track": s["source_track"],
        })

    swaps = detect_id_swaps(per_crop_identity)
    for sw in swaps:
        mutations.append({
            "type": "id_swap",
            "description": f"Swap {sw['track_a']} <-> {sw['track_b']} at frame {sw['crossover_frame']}",
            "tracks": [sw["track_a"], sw["track_b"]],
        })

    merges = detect_fragment_merges(simple_track_info)
    for m in merges:
        mutations.append({
            "type": "fragment_merge",
            "description": f"Merge {m['merge_track']} into {m['keep_track']}",
            "keep_track": m["keep_track"],
            "merge_track": m["merge_track"],
        })

    # Detect outlier keyframe moves
    for rid, info in track_info.items():
        if len(info["identity_counts"]) < 2:
            continue
        majority = info["majority_identity"]
        for frame, ident in info["per_frame"].items():
            if ident != majority:
                # Find target track for this identity
                target = None
                for other_rid, other_info in track_info.items():
                    if other_rid != rid and other_info["majority_identity"] == ident:
                        target = other_rid
                        break
                if target:
                    mutations.append({
                        "type": "keyframe_move",
                        "description": f"Move frame {frame} from {rid} to {target}",
                        "source_track": rid,
                        "target_track": target,
                        "frame": frame,
                    })

    # ID labeling
    identities = {}
    for rid, info in track_info.items():
        identities[rid] = info["majority_identity"]

    # Add id_label mutations
    for rid, ident in identities.items():
        mutations.append({
            "type": "id_label",
            "description": f"Label {rid} as id:{ident}",
            "region_id": rid,
            "identity": ident,
        })

    return {
        "mutations": mutations,
        "identities": identities,
        "n_tracks_after": len(set(identities.values())),
    }


def execute_writeback(
    session,
    annotation_result: List[Dict[str, Any]],
    per_crop_identity: Dict[str, int],
    progress=None,
) -> List[Dict[str, Any]]:
    """Execute all annotation mutations and return the modified result.

    Order: splits → moves → swaps → merges → ID labels
    """
    result = copy.deepcopy(annotation_result)
    track_info = compute_track_identities(per_crop_identity)

    if progress:
        progress.step = "Applying track splits..."
        progress.current = 1
        progress.total = 5

    # 1. Splits
    splits = detect_track_splits(per_crop_identity)
    for s in splits:
        result = apply_track_split(result, s)

    if progress:
        progress.step = "Moving outlier keyframes..."
        progress.current = 2

    # 2. Keyframe moves (outliers)
    for rid, info in track_info.items():
        if len(info["identity_counts"]) < 2:
            continue
        majority = info["majority_identity"]
        for frame, ident in info["per_frame"].items():
            if ident != majority:
                target = None
                for other_rid, other_info in track_info.items():
                    if other_rid != rid and other_info["majority_identity"] == ident:
                        target = other_rid
                        break
                if target:
                    result = apply_keyframe_move(result, {
                        "source_track": rid,
                        "target_track": target,
                        "frame": frame,
                    })

    if progress:
        progress.step = "Applying ID swaps..."
        progress.current = 3

    # 3. Swaps
    swaps = detect_id_swaps(per_crop_identity)
    for sw in swaps:
        result = apply_id_swap(result, sw)

    if progress:
        progress.step = "Merging fragmented tracks..."
        progress.current = 4

    # 4. Fragment merges
    simple_info: Dict[str, Dict[str, Any]] = {}
    for region in result:
        if region.get("type") != "videorectangle":
            continue
        rid = region["id"]
        frames = [int(k["frame"]) for k in region["value"].get("sequence", [])]
        if rid in track_info:
            simple_info[rid] = {
                "frames": frames,
                "identity": track_info[rid]["majority_identity"],
            }

    merges = detect_fragment_merges(simple_info)
    for m in merges:
        result = apply_fragment_merge(result, m)

    if progress:
        progress.step = "Applying ID labels..."
        progress.current = 5

    # 5. ID labels
    identity_map = {rid: info["majority_identity"] for rid, info in track_info.items()}
    result = apply_id_labels(result, identity_map)

    return result
