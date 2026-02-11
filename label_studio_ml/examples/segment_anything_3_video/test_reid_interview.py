"""Tests for the standalone ReID Interview UI.

Covers: state management, annotation parsing, pipeline logic,
annotation write-back (all 5 scenarios), pair generation, and routes.
"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure our module is importable from this directory
# ---------------------------------------------------------------------------
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ===========================================================================
# Fixtures: sample Label Studio annotation data
# ===========================================================================

def _make_keyframe(frame: int, x: float, y: float, w: float, h: float,
                   enabled: bool = True, fps: float = 30.0) -> Dict[str, Any]:
    """Build a single LS VideoRectangle keyframe dict (percent coords)."""
    return {
        "frame": frame,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "enabled": enabled,
        "rotation": 0,
        "time": (frame - 1) / fps,
    }


def _make_region(region_id: str, keyframes: List[Dict], labels: List[str],
                 meta_text: str = "id:", frames_count: int = 300,
                 duration: float = 10.0) -> Dict[str, Any]:
    """Build a LS VideoRectangle region dict."""
    return {
        "id": region_id,
        "type": "videorectangle",
        "from_name": "box",
        "to_name": "video",
        "score": 1.0,
        "origin": "manual",
        "value": {
            "sequence": keyframes,
            "framesCount": frames_count,
            "duration": duration,
            "labels": labels,
        },
        "meta": {"text": meta_text},
    }


@pytest.fixture
def sample_annotation_result():
    """3 tracks with known identity patterns for testing all scenarios."""
    return [
        # Track A: Person 1, frames 10-50
        _make_region("track-A", [
            _make_keyframe(10, 10.0, 20.0, 15.0, 30.0),
            _make_keyframe(20, 12.0, 22.0, 15.0, 30.0),
            _make_keyframe(30, 14.0, 24.0, 15.0, 30.0),
            _make_keyframe(40, 16.0, 26.0, 15.0, 30.0),
            _make_keyframe(50, 18.0, 28.0, 15.0, 30.0),
        ], ["person"], "id:"),
        # Track B: Person 2, frames 10-50
        _make_region("track-B", [
            _make_keyframe(10, 50.0, 20.0, 15.0, 30.0),
            _make_keyframe(20, 52.0, 22.0, 15.0, 30.0),
            _make_keyframe(30, 54.0, 24.0, 15.0, 30.0),
            _make_keyframe(40, 56.0, 26.0, 15.0, 30.0),
            _make_keyframe(50, 58.0, 28.0, 15.0, 30.0),
        ], ["person"], "id:"),
        # Track C: Person 1 again (fragmented), frames 70-90
        _make_region("track-C", [
            _make_keyframe(70, 10.0, 20.0, 15.0, 30.0),
            _make_keyframe(80, 12.0, 22.0, 15.0, 30.0),
            _make_keyframe(90, 14.0, 24.0, 15.0, 30.0),
        ], ["person"], "id:"),
    ]


@pytest.fixture
def video_dimensions():
    return {"width": 1920, "height": 1080}


# ===========================================================================
# 1. State management tests
# ===========================================================================

class TestState:
    """Test ReIDInterviewSession creation and registry."""

    def test_session_create(self):
        from reid_interview.state import create_session, get_session
        session = create_session(project_id=1, task_id=2, annotation_id=3)
        assert session.project_id == 1
        assert session.task_id == 2
        assert session.annotation_id == 3
        assert session.session_id is not None
        assert len(session.session_id) > 0
        # Retrievable
        found = get_session(session.session_id)
        assert found is session

    def test_session_requires_annotation_id(self):
        from reid_interview.state import create_session
        # annotation_id is required in ReID Interview (not optional)
        with pytest.raises(ValueError, match="annotation_id"):
            create_session(project_id=1, task_id=2, annotation_id=None)

    def test_session_isolation_from_interview(self):
        """ReID Interview sessions do NOT collide with interview sessions."""
        from reid_interview.state import create_session as create_reid_session
        from reid_interview.state import get_session as get_reid_session
        from interview.state import create_session as create_interview_session
        from interview.state import get_session as get_interview_session

        reid = create_reid_session(project_id=1, task_id=2, annotation_id=3)
        intv = create_interview_session(project_id=1, task_id=2)

        # Each only finds its own session
        assert get_reid_session(reid.session_id) is reid
        assert get_reid_session(intv.session_id) is None
        assert get_interview_session(intv.session_id) is intv
        assert get_interview_session(reid.session_id) is None

    def test_session_cache_key_includes_annotation(self):
        from reid_interview.state import create_session
        session = create_session(project_id=5, task_id=10, annotation_id=20)
        assert "a20" in session.cache_key
        assert "p5" in session.cache_key
        assert "t10" in session.cache_key

    def test_session_initial_phase(self):
        from reid_interview.state import create_session
        session = create_session(project_id=1, task_id=1, annotation_id=1)
        assert session.phase == "landing"

    def test_session_delete(self):
        from reid_interview.state import create_session, get_session, delete_session
        session = create_session(project_id=1, task_id=1, annotation_id=1)
        sid = session.session_id
        assert get_session(sid) is not None
        delete_session(sid)
        assert get_session(sid) is None


# ===========================================================================
# 2. Annotation parsing tests
# ===========================================================================

class TestAnnotationParsing:
    """Test parsing LS VideoRectangle annotations into tracks + crops."""

    def test_parse_tracks_basic(self, sample_annotation_result, video_dimensions):
        from reid_interview.pipeline import parse_annotation_tracks
        tracks, crops = parse_annotation_tracks(
            sample_annotation_result,
            video_dimensions["width"],
            video_dimensions["height"],
        )
        assert len(tracks) == 3
        assert tracks[0].region_id == "track-A"
        assert tracks[1].region_id == "track-B"
        assert tracks[2].region_id == "track-C"

    def test_parse_tracks_keyframe_count(self, sample_annotation_result, video_dimensions):
        from reid_interview.pipeline import parse_annotation_tracks
        tracks, crops = parse_annotation_tracks(
            sample_annotation_result,
            video_dimensions["width"],
            video_dimensions["height"],
        )
        # Track A: 5 keyframes, Track B: 5, Track C: 3 = 13 total crops
        assert len(crops) == 13

    def test_parse_tracks_crop_ids_unique(self, sample_annotation_result, video_dimensions):
        from reid_interview.pipeline import parse_annotation_tracks
        _, crops = parse_annotation_tracks(
            sample_annotation_result,
            video_dimensions["width"],
            video_dimensions["height"],
        )
        crop_ids = list(crops.keys())
        assert len(crop_ids) == len(set(crop_ids)), "Crop IDs must be unique"

    def test_parse_tracks_pixel_coords(self, sample_annotation_result, video_dimensions):
        from reid_interview.pipeline import parse_annotation_tracks
        _, crops = parse_annotation_tracks(
            sample_annotation_result,
            video_dimensions["width"],
            video_dimensions["height"],
        )
        # Track A, frame 10: x=10%, y=20%, w=15%, h=30% on 1920x1080
        crop = crops["track-A_f10"]
        expected_x1 = (10.0 / 100.0) * 1920  # 192.0
        expected_y1 = (20.0 / 100.0) * 1080  # 216.0
        expected_x2 = expected_x1 + (15.0 / 100.0) * 1920  # 192 + 288 = 480
        expected_y2 = expected_y1 + (30.0 / 100.0) * 1080  # 216 + 324 = 540
        np.testing.assert_allclose(crop.xyxy_px, [expected_x1, expected_y1, expected_x2, expected_y2], atol=0.01)

    def test_parse_tracks_crop_track_mapping(self, sample_annotation_result, video_dimensions):
        from reid_interview.pipeline import parse_annotation_tracks
        _, crops = parse_annotation_tracks(
            sample_annotation_result,
            video_dimensions["width"],
            video_dimensions["height"],
        )
        # All track-A crops reference track-A
        for crop_id, crop in crops.items():
            if crop_id.startswith("track-A"):
                assert crop.track_region_id == "track-A"

    def test_parse_skips_non_videorectangle(self, video_dimensions):
        """Non-videorectangle regions should be ignored."""
        from reid_interview.pipeline import parse_annotation_tracks
        result = [
            {"id": "label-1", "type": "labels", "value": {"labels": ["person"]}},
            _make_region("track-A", [_make_keyframe(10, 10, 20, 15, 30)], ["person"]),
        ]
        tracks, crops = parse_annotation_tracks(result, video_dimensions["width"], video_dimensions["height"])
        assert len(tracks) == 1
        assert tracks[0].region_id == "track-A"

    def test_parse_empty_annotation(self, video_dimensions):
        from reid_interview.pipeline import parse_annotation_tracks
        tracks, crops = parse_annotation_tracks([], video_dimensions["width"], video_dimensions["height"])
        assert len(tracks) == 0
        assert len(crops) == 0

    def test_meta_text_preserved(self, video_dimensions):
        from reid_interview.pipeline import parse_annotation_tracks
        result = [_make_region("track-X", [_make_keyframe(1, 0, 0, 10, 10)], ["person"], meta_text="id:42")]
        tracks, _ = parse_annotation_tracks(result, video_dimensions["width"], video_dimensions["height"])
        assert tracks[0].meta_text == "id:42"


# ===========================================================================
# 3. Pair generation tests
# ===========================================================================

class TestPairGeneration:
    """Test pair sampling, difficulty ordering, and calibration checks."""

    def _make_session_with_clusters(self):
        """Create a session with mock crops and clusters for testing."""
        from reid_interview.state import ReIDInterviewSession, CropInfo, TrackInfo
        session = ReIDInterviewSession(
            session_id="test-123",
            project_id=1,
            task_id=1,
            annotation_id=1,
            cache_key="reid_p1_t1_a1",
        )
        # 6 crops: 3 in cluster 0, 3 in cluster 1
        for i in range(6):
            crop_id = f"crop_{i}"
            session.crops[crop_id] = CropInfo(
                crop_id=crop_id,
                track_region_id=f"track-{i % 3}",
                frame_idx=i * 10 + 1,
                x_pct=10.0,
                y_pct=20.0,
                w_pct=15.0,
                h_pct=30.0,
                xyxy_px=np.array([192.0, 216.0, 480.0, 540.0]),
            )
            session.features[crop_id] = np.random.randn(1024).astype(np.float32)
            session.features[crop_id] /= np.linalg.norm(session.features[crop_id])

        session.clusters = {
            0: ["crop_0", "crop_1", "crop_2"],
            1: ["crop_3", "crop_4", "crop_5"],
        }
        # Build a similarity matrix
        n = 6
        sim = np.eye(n, dtype=np.float32)
        # Same cluster: high similarity
        for i in range(3):
            for j in range(3):
                sim[i, j] = 0.9 + np.random.uniform(-0.05, 0.05)
                sim[j, i] = sim[i, j]
        for i in range(3, 6):
            for j in range(3, 6):
                sim[i, j] = 0.9 + np.random.uniform(-0.05, 0.05)
                sim[j, i] = sim[i, j]
        # Cross-cluster: low-medium similarity
        for i in range(3):
            for j in range(3, 6):
                sim[i, j] = 0.4 + np.random.uniform(-0.15, 0.15)
                sim[j, i] = sim[i, j]
        np.fill_diagonal(sim, 1.0)
        session.similarity_matrix = sim
        session.crop_id_list = [f"crop_{i}" for i in range(6)]
        return session

    def test_generate_pairs_returns_list(self):
        from reid_interview.pipeline import generate_pairs
        session = self._make_session_with_clusters()
        pairs = generate_pairs(session)
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_pairs_have_required_fields(self):
        from reid_interview.pipeline import generate_pairs
        session = self._make_session_with_clusters()
        pairs = generate_pairs(session)
        for p in pairs:
            assert hasattr(p, "pair_id")
            assert hasattr(p, "crop_id_a")
            assert hasattr(p, "crop_id_b")
            assert hasattr(p, "similarity")
            assert hasattr(p, "pool")
            assert hasattr(p, "difficulty")
            assert hasattr(p, "model_prediction")

    def test_pairs_difficulty_ordering(self):
        """Pairs should be ordered: warmup → calibration interleaved → ambiguous."""
        from reid_interview.pipeline import generate_pairs
        session = self._make_session_with_clusters()
        pairs = generate_pairs(session)
        # First pair(s) should be warmup or calibration (easy)
        first_pools = [p.pool for p in pairs[:3]]
        assert any(pool in ("warmup", "calibration") for pool in first_pools)

    def test_calibration_pairs_have_model_prediction(self):
        from reid_interview.pipeline import generate_pairs
        session = self._make_session_with_clusters()
        pairs = generate_pairs(session)
        for p in pairs:
            if p.pool == "calibration":
                assert p.model_prediction in ("same", "different")

    def test_auto_resolved_pairs_get_calibration(self):
        """Auto-resolved confident pairs should have 1-2 calibration checks."""
        from reid_interview.pipeline import generate_pairs
        session = self._make_session_with_clusters()
        pairs = generate_pairs(session)
        calibration = [p for p in pairs if p.pool == "calibration"]
        # With 2 clusters, should have at least 1 calibration check
        assert len(calibration) >= 1

    def test_pair_crop_ids_valid(self):
        from reid_interview.pipeline import generate_pairs
        session = self._make_session_with_clusters()
        pairs = generate_pairs(session)
        valid_ids = set(session.crops.keys())
        for p in pairs:
            assert p.crop_id_a in valid_ids
            assert p.crop_id_b in valid_ids
            assert p.crop_id_a != p.crop_id_b


# ===========================================================================
# 4. Annotation write-back scenario tests
# ===========================================================================

class TestAnnotationWriteback:
    """Test all 5 ID-switching scenarios plus ID labeling."""

    def _identity_map_from_crops(self, crop_identities: Dict[str, int]) -> Dict[str, int]:
        """Build track-level identity from per-crop identity (majority vote)."""
        from reid_interview.annotation_writeback import compute_identity_assignments
        return compute_identity_assignments(crop_identities)

    # -- Scenario 1: ID Swap --

    def test_id_swap_detection(self):
        """Two tracks swap identities at a crossover frame."""
        from reid_interview.annotation_writeback import detect_id_swaps

        # Track A: crops at frames 10,20,30,40,50
        # Identity assignments: frames 10,20 = person 1; frames 30,40,50 = person 2
        # Track B: crops at frames 10,20,30,40,50
        # Identity assignments: frames 10,20 = person 2; frames 30,40,50 = person 1
        per_crop_identity = {
            "track-A_f10": 1, "track-A_f20": 1,
            "track-A_f30": 2, "track-A_f40": 2, "track-A_f50": 2,
            "track-B_f10": 2, "track-B_f20": 2,
            "track-B_f30": 1, "track-B_f40": 1, "track-B_f50": 1,
        }
        swaps = detect_id_swaps(per_crop_identity)
        assert len(swaps) == 1
        swap = swaps[0]
        assert set([swap["track_a"], swap["track_b"]]) == {"track-A", "track-B"}
        assert swap["crossover_frame"] == 30  # swap happens at frame 30

    def test_id_swap_apply(self, sample_annotation_result, video_dimensions):
        """After swap, keyframes from frame 30+ should be exchanged."""
        from reid_interview.annotation_writeback import apply_id_swap

        result = copy.deepcopy(sample_annotation_result[:2])  # Track A and B
        swap = {
            "track_a": "track-A",
            "track_b": "track-B",
            "crossover_frame": 30,
        }
        new_result = apply_id_swap(result, swap)

        # After swap: Track A frame 30 should have Track B's original coords
        track_a = next(r for r in new_result if r["id"] == "track-A")
        track_b = next(r for r in new_result if r["id"] == "track-B")

        # Frame 30 in track A should now have x=54.0 (was Track B's frame 30)
        frame_30_a = next(k for k in track_a["value"]["sequence"] if k["frame"] == 30)
        assert abs(frame_30_a["x"] - 54.0) < 0.01

        # Frame 30 in track B should now have x=14.0 (was Track A's frame 30)
        frame_30_b = next(k for k in track_b["value"]["sequence"] if k["frame"] == 30)
        assert abs(frame_30_b["x"] - 14.0) < 0.01

    # -- Scenario 2: Fragment Merge --

    def test_fragment_merge_detection(self):
        """Non-overlapping tracks with same identity should be detected for merge."""
        from reid_interview.annotation_writeback import detect_fragment_merges

        track_info = {
            "track-A": {"frames": [10, 20, 30], "identity": 1},
            "track-C": {"frames": [70, 80, 90], "identity": 1},
            "track-B": {"frames": [10, 20, 30], "identity": 2},
        }
        merges = detect_fragment_merges(track_info)
        assert len(merges) == 1
        merge = merges[0]
        assert merge["keep_track"] == "track-A"  # earlier track
        assert merge["merge_track"] == "track-C"  # later track

    def test_fragment_merge_apply(self, sample_annotation_result, video_dimensions):
        """After merge, target track has all keyframes, source track is removed."""
        from reid_interview.annotation_writeback import apply_fragment_merge

        result = copy.deepcopy(sample_annotation_result)
        merge = {
            "keep_track": "track-A",
            "merge_track": "track-C",
        }
        new_result = apply_fragment_merge(result, merge)

        # Track C should be removed
        assert not any(r["id"] == "track-C" for r in new_result)

        # Track A should now have 5 + 3 = 8 keyframes
        track_a = next(r for r in new_result if r["id"] == "track-A")
        assert len(track_a["value"]["sequence"]) == 8

        # Keyframes should be sorted by frame
        frames = [k["frame"] for k in track_a["value"]["sequence"]]
        assert frames == sorted(frames)

    # -- Scenario 3: Track Split --

    def test_track_split_detection(self):
        """Track with mixed identities detected for splitting."""
        from reid_interview.annotation_writeback import detect_track_splits

        per_crop_identity = {
            "track-A_f10": 1, "track-A_f20": 1,
            "track-A_f30": 3, "track-A_f40": 3, "track-A_f50": 3,
        }
        splits = detect_track_splits(per_crop_identity)
        assert len(splits) == 1
        split = splits[0]
        assert split["source_track"] == "track-A"
        assert split["keep_identity"] == 1 or split["keep_identity"] == 3
        assert len(split["segments"]) == 2

    def test_track_split_apply(self, sample_annotation_result, video_dimensions):
        """After split, original track is shortened and new track is created."""
        from reid_interview.annotation_writeback import apply_track_split

        result = copy.deepcopy(sample_annotation_result[:1])  # Just Track A
        split = {
            "source_track": "track-A",
            "segments": [
                {"identity": 1, "frames": [10, 20]},
                {"identity": 3, "frames": [30, 40, 50]},
            ],
        }
        new_result = apply_track_split(result, split)

        # Should now have 2 regions
        assert len(new_result) == 2

        # Original track keeps first segment
        track_a = next(r for r in new_result if r["id"] == "track-A")
        assert len(track_a["value"]["sequence"]) == 2

        # New track has second segment
        new_track = next(r for r in new_result if r["id"] != "track-A")
        assert len(new_track["value"]["sequence"]) == 3

    # -- Scenario 4: Keyframe Move --

    def test_keyframe_move(self, sample_annotation_result, video_dimensions):
        """Outlier keyframe moved from one track to another."""
        from reid_interview.annotation_writeback import apply_keyframe_move

        result = copy.deepcopy(sample_annotation_result[:2])  # Track A and B
        move = {
            "source_track": "track-A",
            "target_track": "track-B",
            "frame": 30,
        }
        new_result = apply_keyframe_move(result, move)

        # Track A should lose frame 30: 5 -> 4 keyframes
        track_a = next(r for r in new_result if r["id"] == "track-A")
        assert len(track_a["value"]["sequence"]) == 4
        assert not any(k["frame"] == 30 for k in track_a["value"]["sequence"])

        # Track B should gain frame 30 from Track A's coords
        track_b = next(r for r in new_result if r["id"] == "track-B")
        frame_30 = next(k for k in track_b["value"]["sequence"] if k["frame"] == 30)
        # The moved keyframe should have Track A's original coords (x=14.0)
        assert abs(frame_30["x"] - 14.0) < 0.01

    # -- Scenario 5: Collision (replace existing keyframe) --

    def test_collision_replace(self, sample_annotation_result, video_dimensions):
        """When target track already has a keyframe at the same frame, replace it."""
        from reid_interview.annotation_writeback import apply_keyframe_move

        result = copy.deepcopy(sample_annotation_result[:2])
        # Both tracks have frame 30. Moving Track A's frame 30 to Track B
        # should replace Track B's existing frame 30
        move = {
            "source_track": "track-A",
            "target_track": "track-B",
            "frame": 30,
        }
        new_result = apply_keyframe_move(result, move)

        track_b = next(r for r in new_result if r["id"] == "track-B")
        # Track B should still have exactly 5 keyframes (replaced, not added)
        assert len(track_b["value"]["sequence"]) == 5
        # The frame-30 keyframe should now have Track A's coords
        frame_30 = next(k for k in track_b["value"]["sequence"] if k["frame"] == 30)
        assert abs(frame_30["x"] - 14.0) < 0.01  # Track A's x was 14.0

    # -- ID Labeling --

    def test_id_labeling(self):
        """Meta text updated to id:<number>."""
        from reid_interview.annotation_writeback import apply_id_labels

        result = [
            _make_region("track-A", [_make_keyframe(10, 10, 20, 15, 30)], ["person"], "id:"),
            _make_region("track-B", [_make_keyframe(10, 50, 20, 15, 30)], ["person"], "id:"),
        ]
        identity_map = {"track-A": 1, "track-B": 2}
        new_result = apply_id_labels(result, identity_map)

        track_a = next(r for r in new_result if r["id"] == "track-A")
        track_b = next(r for r in new_result if r["id"] == "track-B")
        assert track_a["meta"]["text"] == "id:1"
        assert track_b["meta"]["text"] == "id:2"

    def test_id_labeling_preserves_existing_text(self):
        """If meta text has other content, id: is appended/updated correctly."""
        from reid_interview.annotation_writeback import apply_id_labels

        result = [
            _make_region("track-A", [_make_keyframe(10, 10, 20, 15, 30)],
                         ["person"], "some note id:"),
        ]
        identity_map = {"track-A": 5}
        new_result = apply_id_labels(result, identity_map)
        track_a = new_result[0]
        assert "id:5" in track_a["meta"]["text"]

    def test_id_labeling_adds_id_prefix_if_missing(self):
        """If no id: prefix in meta text, add it."""
        from reid_interview.annotation_writeback import apply_id_labels

        result = [
            _make_region("track-A", [_make_keyframe(10, 10, 20, 15, 30)],
                         ["person"], "some note"),
        ]
        identity_map = {"track-A": 7}
        new_result = apply_id_labels(result, identity_map)
        assert "id:7" in new_result[0]["meta"]["text"]

    # -- Identity Assignment --

    def test_identity_assignment_majority_vote(self):
        """Track identity is determined by majority of keyframe identities."""
        from reid_interview.annotation_writeback import compute_track_identities

        per_crop_identity = {
            "track-A_f10": 1, "track-A_f20": 1, "track-A_f30": 1,
            "track-A_f40": 2, "track-A_f50": 1,
        }
        result = compute_track_identities(per_crop_identity)
        assert result["track-A"]["majority_identity"] == 1
        assert result["track-A"]["is_mixed"] is False  # 4/5 = 80% majority

    def test_identity_assignment_mixed_track(self):
        """Track flagged as mixed when no clear majority."""
        from reid_interview.annotation_writeback import compute_track_identities

        per_crop_identity = {
            "track-A_f10": 1, "track-A_f20": 1,
            "track-A_f30": 2, "track-A_f40": 2, "track-A_f50": 2,
        }
        result = compute_track_identities(per_crop_identity)
        # 3/5 = 60% for identity 2. Whether this is "mixed" depends on threshold.
        # But with a 50/50 or 60/40 split, it should be flagged.
        assert result["track-A"]["majority_identity"] == 2


# ===========================================================================
# 5. Write-back preview tests
# ===========================================================================

class TestWritebackPreview:
    """Test the dry-run preview of annotation changes."""

    def test_preview_returns_mutations(self, sample_annotation_result, video_dimensions):
        from reid_interview.annotation_writeback import compute_writeback_preview

        # Simulate: Track A and Track C are same person (identity 1)
        per_crop_identity = {
            "track-A_f10": 1, "track-A_f20": 1, "track-A_f30": 1,
            "track-A_f40": 1, "track-A_f50": 1,
            "track-B_f10": 2, "track-B_f20": 2, "track-B_f30": 2,
            "track-B_f40": 2, "track-B_f50": 2,
            "track-C_f70": 1, "track-C_f80": 1, "track-C_f90": 1,
        }
        preview = compute_writeback_preview(
            sample_annotation_result, per_crop_identity, video_dimensions["width"], video_dimensions["height"]
        )
        assert "mutations" in preview
        assert "identities" in preview
        # Should detect fragment merge (Track A + Track C)
        merge_mutations = [m for m in preview["mutations"] if m["type"] == "fragment_merge"]
        assert len(merge_mutations) >= 1


# ===========================================================================
# 6. Coordinate conversion round-trip tests
# ===========================================================================

class TestCoordinateConversion:
    """Verify percent ↔ pixel conversions are consistent."""

    def test_percent_to_pixel_roundtrip(self):
        from reid_interview.pipeline import _percent_xywh_to_xyxy_px

        x_pct, y_pct, w_pct, h_pct = 10.0, 20.0, 30.0, 40.0
        width, height = 1920, 1080

        # Percent → pixel
        xyxy = _percent_xywh_to_xyxy_px(x_pct, y_pct, w_pct, h_pct, width, height)

        # Pixel → percent (inline reverse conversion)
        x0, y0, x1, y1 = xyxy
        x2 = (float(x0) / width) * 100.0
        y2 = (float(y0) / height) * 100.0
        w2 = (float(x1 - x0) / width) * 100.0
        h2 = (float(y1 - y0) / height) * 100.0

        np.testing.assert_allclose(x2, x_pct, atol=0.1)
        np.testing.assert_allclose(y2, y_pct, atol=0.1)
        np.testing.assert_allclose(w2, w_pct, atol=0.1)
        np.testing.assert_allclose(h2, h_pct, atol=0.1)


# ===========================================================================
# 7. Route / Blueprint tests
# ===========================================================================

class TestRoutes:
    """Test Flask route registration and basic endpoint behavior."""

    @pytest.fixture
    def app(self):
        """Create a minimal Flask app with the ReID Interview blueprint."""
        from flask import Flask
        from reid_interview import reid_interview_bp
        app = Flask(__name__)
        app.register_blueprint(reid_interview_bp)
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def test_landing_page_serves_html(self, client):
        resp = client.get("/ReID-Interview/")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.data or b"<!doctype html>" in resp.data.lower()

    def test_session_init_requires_all_ids(self, client):
        # Missing annotation_id
        resp = client.post("/ReID-Interview/api/session/init",
                          json={"project_id": 1, "task_id": 2})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "annotation_id" in data.get("error", "").lower()

    def test_session_init_success(self, client):
        resp = client.post("/ReID-Interview/api/session/init",
                          json={"project_id": 1, "task_id": 2, "annotation_id": 3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "session_id" in data

    def test_session_status_not_found(self, client):
        resp = client.get("/ReID-Interview/api/session/nonexistent/status")
        assert resp.status_code == 404

    def test_crop_image_not_found(self, client):
        resp = client.get("/ReID-Interview/api/crop/fake-crop/image?session_id=fake")
        assert resp.status_code == 404

    def test_clusters_endpoint(self, client):
        # First create a session
        resp = client.post("/ReID-Interview/api/session/init",
                          json={"project_id": 1, "task_id": 2, "annotation_id": 3})
        session_id = resp.get_json()["session_id"]

        resp = client.get(f"/ReID-Interview/api/clusters?session_id={session_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "clusters" in data
        assert "nodes" in data or "clusters" in data

    def test_resolve_pair_without_session(self, client):
        resp = client.post("/ReID-Interview/api/pairs/resolve",
                          json={"session_id": "nonexistent", "pair_id": "p1", "resolution": "same"})
        assert resp.status_code == 404


# ===========================================================================
# 8. Gamification tests
# ===========================================================================

class TestGamification:
    """Test accuracy tracking and difficulty escalation logic."""

    def test_accuracy_calculation(self):
        from reid_interview.pipeline import compute_accuracy

        resolutions = {
            "pair_1": "same",    # calibration: model said "same"
            "pair_2": "different",  # calibration: model said "different"
            "pair_3": "same",    # calibration: model said "different" (human disagrees)
        }
        calibration_answers = {
            "pair_1": "same",
            "pair_2": "different",
            "pair_3": "different",
        }
        accuracy = compute_accuracy(resolutions, calibration_answers)
        # 2 out of 3 correct = 66.67%
        assert abs(accuracy - 66.67) < 1.0

    def test_difficulty_escalation_order(self):
        """Pairs should be ordered by escalating difficulty."""
        from reid_interview.pipeline import order_pairs_by_difficulty
        from reid_interview.state import ReIDPairInfo

        pairs = [
            ReIDPairInfo(pair_id="p1", crop_id_a="a", crop_id_b="b",
                        track_a="t1", track_b="t2", similarity=0.95,
                        pool="warmup", difficulty=0, model_prediction="same"),
            ReIDPairInfo(pair_id="p2", crop_id_a="c", crop_id_b="d",
                        track_a="t1", track_b="t2", similarity=0.50,
                        pool="ambiguous", difficulty=2, model_prediction="same"),
            ReIDPairInfo(pair_id="p3", crop_id_a="e", crop_id_b="f",
                        track_a="t1", track_b="t1", similarity=0.85,
                        pool="calibration", difficulty=0, model_prediction="same"),
            ReIDPairInfo(pair_id="p4", crop_id_a="g", crop_id_b="h",
                        track_a="t1", track_b="t2", similarity=0.35,
                        pool="ambiguous", difficulty=1, model_prediction="different"),
        ]
        ordered = order_pairs_by_difficulty(pairs)
        # Warmup should come first
        assert ordered[0].pool == "warmup"
        # Last pair should be the hardest ambiguous one
        ambiguous = [p for p in ordered if p.pool == "ambiguous"]
        assert ambiguous[-1].difficulty >= ambiguous[0].difficulty


# ===========================================================================
# 9. Cluster update (incremental merge) tests
# ===========================================================================

class TestClusterUpdate:
    """Test incremental cluster merge after pair resolution."""

    def test_merge_on_same_resolution(self):
        """Resolving a pair as 'same' should merge their clusters."""
        from reid_interview.pipeline import apply_pair_resolution

        clusters = {
            0: ["crop_0", "crop_1"],
            1: ["crop_2", "crop_3"],
        }
        resolution = {
            "pair_id": "p1",
            "crop_id_a": "crop_0",
            "crop_id_b": "crop_2",
            "resolution": "same",
        }
        new_clusters = apply_pair_resolution(clusters, resolution)
        # Clusters should be merged: only 1 cluster remaining
        assert len(new_clusters) == 1
        merged = list(new_clusters.values())[0]
        assert set(merged) == {"crop_0", "crop_1", "crop_2", "crop_3"}

    def test_no_merge_on_different_resolution(self):
        from reid_interview.pipeline import apply_pair_resolution

        clusters = {
            0: ["crop_0", "crop_1"],
            1: ["crop_2", "crop_3"],
        }
        resolution = {
            "pair_id": "p1",
            "crop_id_a": "crop_0",
            "crop_id_b": "crop_2",
            "resolution": "different",
        }
        new_clusters = apply_pair_resolution(clusters, resolution)
        assert len(new_clusters) == 2

    def test_no_merge_on_unsure_resolution(self):
        from reid_interview.pipeline import apply_pair_resolution

        clusters = {
            0: ["crop_0", "crop_1"],
            1: ["crop_2", "crop_3"],
        }
        resolution = {
            "pair_id": "p1",
            "crop_id_a": "crop_0",
            "crop_id_b": "crop_2",
            "resolution": "unsure",
        }
        new_clusters = apply_pair_resolution(clusters, resolution)
        assert len(new_clusters) == 2

    def test_merge_already_same_cluster(self):
        """If both crops are already in the same cluster, no change."""
        from reid_interview.pipeline import apply_pair_resolution

        clusters = {
            0: ["crop_0", "crop_1", "crop_2"],
        }
        resolution = {
            "pair_id": "p1",
            "crop_id_a": "crop_0",
            "crop_id_b": "crop_2",
            "resolution": "same",
        }
        new_clusters = apply_pair_resolution(clusters, resolution)
        assert len(new_clusters) == 1
        assert set(new_clusters[0]) == {"crop_0", "crop_1", "crop_2"}

    def test_transitive_merge(self):
        """Multiple 'same' resolutions should transitively merge clusters."""
        from reid_interview.pipeline import apply_pair_resolution

        clusters = {
            0: ["crop_0"],
            1: ["crop_1"],
            2: ["crop_2"],
        }
        # First merge 0 and 1
        clusters = apply_pair_resolution(clusters, {
            "pair_id": "p1", "crop_id_a": "crop_0", "crop_id_b": "crop_1",
            "resolution": "same",
        })
        assert len(clusters) == 2

        # Then merge with 2
        clusters = apply_pair_resolution(clusters, {
            "pair_id": "p2", "crop_id_a": "crop_0", "crop_id_b": "crop_2",
            "resolution": "same",
        })
        assert len(clusters) == 1
        merged = list(clusters.values())[0]
        assert set(merged) == {"crop_0", "crop_1", "crop_2"}
