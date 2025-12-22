import numpy as np

from typing import List, Dict


def get_label_map(labels: List[str]) -> Dict:
    """
    Generate a label map from a list of labels.
    Args:
        labels: List of label names
    Returns:
        label_map: Dictionary mapping label names to indices
    """
    return {label: idx for idx, label in enumerate(sorted(labels))}


def convert_timelinelabels_to_probs(
    regions: List[Dict], label_map: Dict[str, int], max_frame=None
) -> (np.ndarray, Dict):
    """Generated numpy array with shape (num_frames, num_labels) and label mapping from timeline regions.
    Args:
        regions: List of timeline regions from annotation
        label_map: Dictionary mapping label names to indices
        max_frame: Maximum frame number in video
    Returns:
        labels_array: Numpy array with shape (num_frames, num_labels)
        used_labels: Labels that were used in the regions
    """
    # Step 1: Collect all unique labels and map them to an index
    used_labels = set()

    # Step 1: Identify all unique labels
    for region in regions:
        labels = region["value"]["timelinelabels"]
        used_labels.update(labels)

    # Step 2: Find the maximum frame index to define the array's X-axis size
    if max_frame is None:
        max_frame = 0
        for region in regions:
            for r in region["value"]["ranges"]:
                max_frame = max(max_frame, r["end"])

    # Step 3: Create a numpy array with shape (num_frames, num_labels)
    # Initialize it with zeros (no label assigned)
    num_labels = len(label_map)
    labels_array = np.zeros((max_frame, num_labels), dtype=int)

    # Step 4: Populate the array with labels based on frame ranges
    for region in regions:
        start_frame = region["value"]["ranges"][0]["start"] - 1
        end_frame = region["value"]["ranges"][0]["end"]
        label_name = region["value"]["timelinelabels"][0]
        label_idx = label_map[label_name]

        # Set the corresponding frames to 1 for the given label
        labels_array[start_frame:end_frame, label_idx] = 1

    return labels_array, used_labels


def convert_probs_to_timelinelabels(
    probs, label_mapping, from_name, score_threshold=0.5
) -> List[Dict]:
    """
    Generate timeline labels regions based on the given probabilities and label mapping.

    Args:
    - probs: 2D numpy array or tensor of probabilities (shape: [num_frames, num_labels])
    - label_mapping: dict mapping label names to indices in the probs array
    - from_name: name of the control tag in the Label Studio configuration
    - score_threshold: threshold above which a label is considered active for a frame

    Returns:
    - regions: List of regions in Label Studio format
    """

    # Initialize a dictionary to keep track of ongoing segments for each label
    regions, added = [], 0
    ongoing_segments = {label: {} for label in label_mapping}

    num_frames = len(probs)  # Number of frames
    if num_frames == 0:
        return regions

    # Iterate through each frame
    for i in range(num_frames):
        # Get probabilities for the current frame
        frame_probs = probs[i]

        # Iterate through each label
        for label, label_idx in label_mapping.items():
            prob = frame_probs[label_idx]
            segment = ongoing_segments[label]

            # Check if the probability exceeds the threshold
            if prob >= score_threshold:
                # Start a new segment if none exists
                if not segment:
                    segment["idx"] = added
                    segment["start"] = i + 1
                    segment["label"] = label
                    segment["score"] = float(prob)
                    segment["from_name"] = from_name
                    added += 1
                else:
                    segment["score"] += float(prob)
            else:
                # Close the ongoing segment if probability falls below the threshold
                if segment:
                    segment["end"] = i
                    segment["score"] /= i - (segment["start"] - 1)
                    regions.append(create_timeline_region(**segment))
                    segment.clear()

    # Close any ongoing segments at the end of the video
    for label, segment in ongoing_segments.items():
        if segment:
            segment["end"] = num_frames
            segment["score"] /= num_frames - (segment["start"] - 1)
            regions.append(create_timeline_region(**segment))

    return regions


def create_timeline_region(idx, start, end, label, score, from_name):
    """
    Helper function to add a timeline region to the timeline_labels list.
    """
    return {
        "id": f"{idx}_{start}_{end}",
        "type": "timelinelabels",
        "value": {
            "ranges": [{"start": start, "end": end}],
            "timelinelabels": [label],
        },
        "to_name": "video",  # Customize if needed
        "from_name": from_name,  # Customize if needed
        "score": score,
    }
