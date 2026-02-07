"""
Tests for tracking fixes in initial_seeding_video_boxes.py and
initial_seeding_video_boxes_manual_merge.py.

Verifies:
1. Seed frame NOT double-counted (only in forward, excluded from backward)
2. Seed frame uses original annotation box (not model re-prediction)
3. Scores use object_score_logits (not binarized mask mean of 1.0)
4. Single-frame window returns original seed box (not empty)
5. object_score_logits early termination when score drops below threshold
6. Detection oracle cross-check truncates tracklets on ID switch

Run: pytest test_tracking_fixes.py -v
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures: mock SAM3 model outputs
# ---------------------------------------------------------------------------

@dataclass
class MockTrackerOutput:
    """Simulates Sam3TrackerVideoSegmentationOutput."""
    frame_idx: int
    pred_masks: torch.Tensor  # (1, 1, H, W)
    object_ids: List[int] = field(default_factory=lambda: [0])
    object_score_logits: Optional[torch.Tensor] = None  # (1,)


def _make_mask_with_box(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    """Create a binary mask tensor with a filled rectangle."""
    mask = torch.zeros(1, 1, h, w, dtype=torch.float32)
    mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def _make_empty_mask(h: int, w: int) -> torch.Tensor:
    """Create an all-zeros mask (object disappeared)."""
    return torch.zeros(1, 1, h, w, dtype=torch.float32)


@dataclass
class MockVideoOutput:
    """Simulates Sam3VideoSegmentationOutput for detection oracle."""
    frame_idx: int
    object_ids: List[int] = field(default_factory=list)
    obj_id_to_mask: Dict = field(default_factory=dict)
    obj_id_to_score: Dict = field(default_factory=dict)
    removed_obj_ids: set = field(default_factory=set)
    suppressed_obj_ids: set = field(default_factory=set)


@pytest.fixture
def mock_tracker():
    """Mock Sam3TrackerVideoModel + Sam3TrackerVideoProcessor."""
    model = MagicMock()
    processor = MagicMock()

    # processor(images=...) returns mock with original_sizes and pixel_values
    mock_inputs = MagicMock()
    mock_inputs.original_sizes = [(100, 100)]
    mock_inputs.pixel_values = [torch.zeros(3, 100, 100)]
    processor.return_value = mock_inputs

    # init_video_session returns mock session
    processor.init_video_session.return_value = MagicMock()
    processor.add_inputs_to_inference_session.return_value = None

    # post_process_masks just returns the masks as-is (already at original size)
    def passthrough_masks(masks_list, original_sizes=None, binarize=True):
        return [m.squeeze(0) for m in masks_list]  # remove batch dim
    processor.post_process_masks.side_effect = passthrough_masks

    return model, processor


@pytest.fixture
def five_frame_pil_images():
    """5 dummy 100x100 PIL images."""
    from PIL import Image
    return [Image.new("RGB", (100, 100), color=(i * 50, 0, 0)) for i in range(5)]


# ===========================================================================
# TEST GROUP 1: Seed frame double-counting
# ===========================================================================

class TestSeedFrameDoubleCounting:
    """After fix: seed frame should appear in forward tracklet (with original box)
    but NOT in backward tracklet's model predictions."""

    def test_forward_includes_seed_frame(self, mock_tracker, five_frame_pil_images):
        """Forward tracklet should include the seed frame."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2  # middle frame

        # Model yields frames 0-4, forward from seed=2 means frames 2,3,4
        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),  # high confidence
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},  # session -> global
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # Seed frame (global=102) should be in forward results
        assert 102 in fwd_boxes, "Seed frame should be included in forward tracklet"
        # Frames after seed should also be present
        assert 103 in fwd_boxes
        assert 104 in fwd_boxes
        # Frames before seed should NOT be present
        assert 100 not in fwd_boxes
        assert 101 not in fwd_boxes

    def test_backward_excludes_seed_frame(self, mock_tracker, five_frame_pil_images):
        """Backward tracklet should NOT include the seed frame (forward owns it)."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2

        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_backward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # Seed frame (global=102) should NOT be in backward results
        assert 102 not in bwd_boxes, "Seed frame should be excluded from backward tracklet"
        # Frames before seed should be present
        assert 100 in bwd_boxes
        assert 101 in bwd_boxes
        # Frames after seed should NOT be present
        assert 103 not in bwd_boxes
        assert 104 not in bwd_boxes

    def test_boxes_py_forward_includes_seed(self, mock_tracker, five_frame_pil_images):
        """Same test for initial_seeding_video_boxes.py's forward function."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2

        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        assert 102 in fwd_boxes, "Seed frame should be included in forward tracklet"
        assert 100 not in fwd_boxes
        assert 101 not in fwd_boxes

    def test_boxes_py_backward_excludes_seed(self, mock_tracker, five_frame_pil_images):
        """Same test for initial_seeding_video_boxes.py's backward function."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2

        outputs = [
            MockTrackerOutput(
                frame_idx=i,
                pred_masks=_make_mask_with_box(100, 100, 10+i, 10+i, 50+i, 50+i),
                object_score_logits=torch.tensor([2.0]),
            )
            for i in range(5)
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes import _generate_backward_tracklet_sam3

        with patch("initial_seeding_video_boxes.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        assert 102 not in bwd_boxes, "Seed frame should be excluded from backward tracklet"
        assert 100 in bwd_boxes
        assert 101 in bwd_boxes


# ===========================================================================
# TEST GROUP 2: Scores from object_score_logits (not always 1.0)
# ===========================================================================

class TestScoreExtraction:
    """After fix: scores should come from object_score_logits, not binarized mask mean."""

    def test_manual_merge_scores_vary(self, mock_tracker, five_frame_pil_images):
        """manual_merge forward tracklet scores should reflect object_score_logits."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        # Decreasing confidence: 0.95, 0.80, 0.60
        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([3.0]),  # sigmoid ~ 0.95
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([1.4]),  # sigmoid ~ 0.80
            ),
            MockTrackerOutput(
                frame_idx=4,
                pred_masks=_make_mask_with_box(100, 100, 14, 14, 54, 54),
                object_score_logits=torch.tensor([0.4]),  # sigmoid ~ 0.60
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # Scores should NOT all be 1.0
        score_values = list(fwd_scores.values())
        assert len(score_values) > 0, "Should have scores"
        assert not all(s == 1.0 for s in score_values), \
            f"Scores should not all be 1.0, got {score_values}"
        # Scores should be decreasing (following logits)
        if len(score_values) >= 2:
            assert score_values[0] > score_values[-1], \
                f"First score should be higher than last: {score_values}"

    def test_manual_merge_mask_to_xyxy_returns_logit_score(self):
        """_mask_to_xyxy should use object_score_logits when provided."""
        from initial_seeding_video_boxes_manual_merge import _mask_to_xyxy

        mask = _make_mask_with_box(100, 100, 10, 10, 50, 50).squeeze(0)  # (1, H, W)
        logits = torch.tensor([1.5])  # sigmoid ~ 0.82

        box, score = _mask_to_xyxy(mask, object_score_logits=logits)

        assert box is not None
        assert score is not None
        assert score != 1.0, f"Score should not be 1.0 (binarized mask mean), got {score}"
        assert 0.8 < score < 0.85, f"Score should be sigmoid(1.5) ~ 0.82, got {score}"


# ===========================================================================
# TEST GROUP 3: Single-frame window
# ===========================================================================

class TestSingleFrameWindow:
    """When win_len == 1 (seed frame is first AND last), both fwd and bwd
    edge guards fire. The seed frame should still get the original annotation box."""

    def test_single_frame_returns_seed_box_manual_merge(self, mock_tracker):
        """manual_merge: single-frame window should yield seed box at seed frame."""
        from PIL import Image
        frames = [Image.new("RGB", (100, 100))]

        from initial_seeding_video_boxes_manual_merge import (
            _generate_forward_tracklet_sam3,
            _generate_backward_tracklet_sam3,
        )

        model, processor = mock_tracker

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={0: 500},
                seed_session_idx=0,
                seed_box_xyxy=np.array([20, 20, 60, 60], dtype=np.float32),
            )
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={0: 500},
                seed_session_idx=0,
                seed_box_xyxy=np.array([20, 20, 60, 60], dtype=np.float32),
            )

        # At least one of fwd/bwd should contain the seed frame with original box
        all_boxes = {**fwd_boxes, **bwd_boxes}
        assert 500 in all_boxes, \
            "Seed frame should have a box even with single-frame window"
        np.testing.assert_array_almost_equal(
            all_boxes[500], [20, 20, 60, 60],
            err_msg="Single-frame window should use original annotation box",
        )


# ===========================================================================
# TEST GROUP 4: object_score_logits early termination
# ===========================================================================

class TestEarlyTermination:
    """Layer 1: When object_score_logits drops below threshold, tracking should stop."""

    def test_forward_stops_on_low_logits(self, mock_tracker, five_frame_pil_images):
        """Forward tracking should terminate when object_score_logits indicates disappearance."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        # Frame 2: high confidence, Frame 3: high, Frame 4: very low (disappeared)
        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([3.0]),  # sigmoid ~ 0.95
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([2.0]),  # sigmoid ~ 0.88
            ),
            MockTrackerOutput(
                frame_idx=4,
                pred_masks=_make_empty_mask(100, 100),
                object_score_logits=torch.tensor([-3.0]),  # sigmoid ~ 0.05 (disappeared!)
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                score_threshold=0.5,
            )

        # Frame 104 (session=4) should NOT be in results (score below threshold)
        assert 104 not in fwd_boxes, \
            "Frame with low object_score_logits should be excluded"
        # Frames 102, 103 should be present
        assert 102 in fwd_boxes or 103 in fwd_boxes, \
            "High-confidence frames should be included"

    def test_backward_stops_on_low_logits(self, mock_tracker, five_frame_pil_images):
        """Backward tracking should terminate when object_score_logits indicates disappearance."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        # Backward from seed=2: frames 1 (ok), 0 (disappeared)
        outputs = [
            MockTrackerOutput(
                frame_idx=1,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([2.0]),
            ),
            MockTrackerOutput(
                frame_idx=0,
                pred_masks=_make_empty_mask(100, 100),
                object_score_logits=torch.tensor([-3.0]),  # disappeared
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_backward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
                score_threshold=0.5,
            )

        # Frame 100 (session=0) should NOT be in results
        assert 100 not in bwd_boxes, \
            "Frame with low object_score_logits should be excluded from backward"
        # Frame 101 should be present
        assert 101 in bwd_boxes


# ===========================================================================
# TEST GROUP 5: Seed frame uses original annotation box
# ===========================================================================

class TestSeedBoxPreservation:
    """Seed frame should use the original annotation box, not model re-prediction."""

    def test_forward_seed_frame_uses_original_box(self, mock_tracker, five_frame_pil_images):
        """At seed frame, the box should be the original annotation, not mask-derived."""
        model, processor = mock_tracker
        frames = five_frame_pil_images
        seed_idx = 2
        original_box = np.array([25, 25, 75, 75], dtype=np.float32)

        # Model predicts a DIFFERENT box at the seed frame (mask-derived would be [10,10,50,50])
        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),  # different!
                object_score_logits=torch.tensor([3.0]),
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([2.5]),
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes_manual_merge import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes_manual_merge.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=seed_idx,
                seed_box_xyxy=original_box,
            )

        # Seed frame box should be the ORIGINAL, not the model-derived [10,10,51,51]
        assert 102 in fwd_boxes
        np.testing.assert_array_almost_equal(
            fwd_boxes[102], original_box,
            err_msg="Seed frame should use original annotation box, not model re-prediction",
        )


# ===========================================================================
# TEST GROUP 6: Detection oracle (Layer 2)
# ===========================================================================

class TestDetectionOracle:
    """Layer 2: Sam3VideoModel oracle should detect ID switches and truncate tracklets."""

    def test_oracle_validates_tracker_output(self):
        """Oracle should flag frames where tracker box doesn't overlap any detected person."""
        # This tests the oracle cross-check function that will be added
        # Import will fail until implementation is done
        try:
            from initial_seeding_video_boxes_manual_merge import _oracle_validate_tracklet
        except ImportError:
            pytest.skip("_oracle_validate_tracklet not yet implemented")

        # Tracker tracked person at these locations
        tracker_boxes = {
            100: np.array([10, 10, 50, 50], dtype=np.float32),  # correct
            101: np.array([12, 12, 52, 52], dtype=np.float32),  # correct
            102: np.array([200, 200, 280, 280], dtype=np.float32),  # ID SWITCH!
            103: np.array([205, 205, 285, 285], dtype=np.float32),  # still wrong
        }

        # Oracle detections: person is at [10-55, 10-55] range on all frames
        oracle_detections = {
            100: [np.array([10, 10, 55, 55], dtype=np.float32)],
            101: [np.array([12, 12, 55, 55], dtype=np.float32)],
            102: [np.array([14, 14, 56, 56], dtype=np.float32)],  # person still here
            103: [np.array([16, 16, 58, 58], dtype=np.float32)],
        }

        validated = _oracle_validate_tracklet(
            tracker_boxes=tracker_boxes,
            oracle_detections=oracle_detections,
            iou_threshold=0.3,
        )

        # Frames 100, 101 should pass (tracker overlaps with detection)
        assert 100 in validated
        assert 101 in validated
        # Frames 102, 103 should be removed (tracker box at [200,200] doesn't
        # overlap with detection at [14,14,56,56])
        assert 102 not in validated, "ID-switched frame should be removed by oracle"
        assert 103 not in validated, "Post-switch frame should also be removed"

    def test_oracle_allows_size_variation(self):
        """Oracle should allow reasonable size variation (not flag normal tracking drift)."""
        try:
            from initial_seeding_video_boxes_manual_merge import _oracle_validate_tracklet
        except ImportError:
            pytest.skip("_oracle_validate_tracklet not yet implemented")

        # Tracker boxes grow slightly (normal tracking behavior)
        tracker_boxes = {
            100: np.array([10, 10, 50, 50], dtype=np.float32),
            101: np.array([9, 9, 52, 52], dtype=np.float32),  # slightly larger
            102: np.array([8, 8, 54, 54], dtype=np.float32),  # a bit more
        }

        oracle_detections = {
            100: [np.array([10, 10, 50, 50], dtype=np.float32)],
            101: [np.array([10, 10, 51, 51], dtype=np.float32)],
            102: [np.array([9, 9, 53, 53], dtype=np.float32)],
        }

        validated = _oracle_validate_tracklet(
            tracker_boxes=tracker_boxes,
            oracle_detections=oracle_detections,
            iou_threshold=0.3,
        )

        # All frames should pass (reasonable IoU overlap)
        assert len(validated) == 3, "Normal tracking drift should not be flagged"


# ===========================================================================
# TEST GROUP 7: _resolve_frame_boxes with varying scores
# ===========================================================================

class TestResolveFrameBoxes:
    """_resolve_frame_boxes should weight by real scores, not degenerate uniform."""

    def test_high_score_dominates(self):
        """With real scores, higher-scored box should dominate in weighted mode."""
        from initial_seeding_video_boxes_manual_merge import _resolve_frame_boxes

        high_score_box = np.array([10, 10, 50, 50], dtype=np.float32)
        low_score_box = np.array([30, 30, 70, 70], dtype=np.float32)

        candidates = [
            (high_score_box, 0.95),
            (low_score_box, 0.10),
        ]

        result = _resolve_frame_boxes(candidates, iou_threshold=0.0, mode="weighted")

        assert result is not None
        # Result should be much closer to high_score_box than low_score_box
        dist_to_high = np.linalg.norm(result - high_score_box)
        dist_to_low = np.linalg.norm(result - low_score_box)
        assert dist_to_high < dist_to_low, \
            f"Result should be closer to high-score box: dist_high={dist_to_high:.2f}, dist_low={dist_to_low:.2f}"

    def test_equal_scores_average(self):
        """With equal scores, result should be midpoint."""
        from initial_seeding_video_boxes_manual_merge import _resolve_frame_boxes

        box_a = np.array([0, 0, 40, 40], dtype=np.float32)
        box_b = np.array([20, 20, 60, 60], dtype=np.float32)

        candidates = [(box_a, 0.5), (box_b, 0.5)]
        result = _resolve_frame_boxes(candidates, iou_threshold=0.0, mode="weighted")

        assert result is not None
        expected = np.array([10, 10, 50, 50], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_winner_mode_picks_highest_score(self):
        """Winner mode should pick the highest-scored box."""
        from initial_seeding_video_boxes_manual_merge import _resolve_frame_boxes

        candidates = [
            (np.array([0, 0, 10, 10], dtype=np.float32), 0.3),
            (np.array([50, 50, 90, 90], dtype=np.float32), 0.9),
        ]

        result = _resolve_frame_boxes(candidates, mode="winner")
        assert result is not None
        np.testing.assert_array_almost_equal(result, [50, 50, 90, 90])


# ===========================================================================
# TEST GROUP 8: boxes.py also returns scores from tracklet generators
# ===========================================================================

class TestBoxesPyScores:
    """After fix: boxes.py forward/backward functions should also return scores."""

    def test_forward_returns_scores(self, mock_tracker, five_frame_pil_images):
        """boxes.py _generate_forward_tracklet_sam3 should return (boxes, scores) tuple."""
        model, processor = mock_tracker
        frames = five_frame_pil_images

        outputs = [
            MockTrackerOutput(
                frame_idx=2,
                pred_masks=_make_mask_with_box(100, 100, 10, 10, 50, 50),
                object_score_logits=torch.tensor([2.0]),
            ),
            MockTrackerOutput(
                frame_idx=3,
                pred_masks=_make_mask_with_box(100, 100, 12, 12, 52, 52),
                object_score_logits=torch.tensor([1.5]),
            ),
        ]
        model.propagate_in_video_iterator.return_value = iter(outputs)

        from initial_seeding_video_boxes import _generate_forward_tracklet_sam3

        with patch("initial_seeding_video_boxes.base._get_sam3_tracker_model",
                    return_value=(model, processor)):
            result = _generate_forward_tracklet_sam3(
                frames_list=frames,
                frame_idx_map={i: i + 100 for i in range(5)},
                seed_session_idx=2,
                seed_box_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
            )

        # After fix, should return (boxes_dict, scores_dict) tuple
        assert isinstance(result, tuple), \
            "boxes.py forward should return (boxes, scores) tuple"
        fwd_boxes, fwd_scores = result
        assert isinstance(fwd_boxes, dict)
        assert isinstance(fwd_scores, dict)
        assert len(fwd_scores) > 0, "Should have scores"
