import numpy as np
from typing import List, Dict


def convert_timelinelabels_to_probs(regions: List[Dict], max_frame=None) -> (np.ndarray, Dict):
    """ Generated numpy array with shape (num_frames, num_labels) and label mapping from timeline regions.
    Args:
        regions: List of timeline regions from annotation
        max_frame: Maximum frame number in video
    Returns:
        labels_array: Numpy array with shape (num_frames, num_labels)
        label_mapping: Mapping of label names to indices
    """
    # Step 1: Collect all unique labels and map them to an index
    all_labels = set()

    # Identify all unique labels
    for region in regions:
        labels = region['value']['timelinelabels']
        all_labels.update(labels)

    # Assign each label an index for the Y-axis in the output array
    label_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}

    # Step 2: Find the maximum frame index to define the array's X-axis size
    if max_frame is None:
        max_frame = 0
        for region in regions:
            for r in region['value']['ranges']:
                max_frame = max(max_frame, r['end']+1)

    # Step 3: Create a numpy array with shape (num_frames, num_labels)
    # Initialize it with zeros (no label assigned)
    num_labels = len(label_mapping)
    labels_array = np.zeros((max_frame, num_labels), dtype=int)

    # Step 4: Populate the array with labels based on frame ranges
    for region in regions:
        start_frame = region['value']['ranges'][0]['start']
        end_frame = region['value']['ranges'][0]['end']
        end_frame = end_frame + (1 if end_frame < max_frame else 0)  # close the gap
        label_name = region['value']['timelinelabels'][0]
        label_idx = label_mapping[label_name]

        # Set the corresponding frames to 1 for the given label
        labels_array[start_frame:end_frame, label_idx] = 1

    return labels_array, label_mapping


def convert_probs_to_timelinelabels(probs, label_mapping, score_threshold=0.5) -> List[Dict]:
    """
    Generate timeline labels regions based on the given probabilities and label mapping.

    Args:
    - probs: 2D numpy array or tensor of probabilities (shape: [num_frames, num_labels])
    - label_mapping: dict mapping label names to indices in the probs array
    - score_threshold: threshold above which a label is considered active for a frame

    Returns:
    - timeline_regions: List of regions in Label Studio format
    """

    # Initialize a dictionary to keep track of ongoing segments for each label
    timeline_regions = []
    ongoing_segments = {label: {} for label in label_mapping}

    num_frames = len(probs)  # Number of frames
    if num_frames == 0:
        return timeline_regions
    num_labels = len(probs[0])  # Number of labels

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
                    segment["start"] = i
            else:
                # Close the ongoing segment if probability falls below the threshold
                if segment:
                    add_timeline_region(i, label, segment, timeline_regions)
                    segment.clear()

    # Close any ongoing segments at the end of the video
    for label, segment in ongoing_segments.items():
        if segment:
            add_timeline_region(num_frames, label, segment, timeline_regions)

    return timeline_regions


def add_timeline_region(i, label, segment, timeline_labels):
    """
    Helper function to add a timeline region to the timeline_labels list.
    """
    timeline_labels.append({
        "id": f"{segment['start']}_{i}",
        "type": "timelinelabels",
        "value": {
            "ranges": [{"start": segment['start'], "end": i}],
            "timelinelabels": [label]
        },
        "to_name": "video",  # Customize if needed
        "from_name": "videoLabels"  # Customize if needed
    })
    return timeline_labels
