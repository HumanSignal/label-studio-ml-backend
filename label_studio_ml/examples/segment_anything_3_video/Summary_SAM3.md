Computer Vision Project: Intent and Requirements
This project focuses on multi-object video segmentation and tracking for long video sequences, typically more than 5000 frames at 720p resolution. The system adapts its processing strategy dynamically and uses SAM3-based computer vision models to stay efficient and accurate over long sequences.

Section 1: Input Types and Model Selection
The system automatically chooses between two SAM3 model variants depending on user inputs and the meaning of the hints flag, which specifies how bounding box prompts should be interpreted.

A. Generic Video Segmentation Model (Sam3VideoModel)
Purpose: Versatile segmentation for many prompt types.
Text-based detection: A textual prompt such as prompt_text="person" finds and segments all instances of that class throughout the video or specified chunk.
Guided detection with exemplars: When a bounding box prompt (box_prompt) is given and hints=true, the box serves as an example rather than a strict target. The model uses the box and optionally a text prompt to find visually similar objects.
Example: prompt_text="dog" plus a bounding box around one dog will help find all similar dogs in the video.

B. Dedicated Object Tracker (Sam3TrackerVideoModel)
Purpose: Maintain a single object’s identity and segmentation mask across time.
Strict instance tracking: When hints=false, the bounding box defines an exact target instance. The model locks onto this instance and propagates its segmentation mask and identity across all frames.

Section 2: Processing Strategies for Long Videos
The system supports two main processing modes to balance memory constraints and accuracy for long videos.

A. Streaming Mode (Default)
Goal: Real-time or near-real-time processing for very large videos.
Mechanism: Frames are processed sequentially, one by one, minimizing memory use.
Trade-off: Extremely memory-efficient but slightly less accurate because each frame is processed without access to future frames.

B. Chunked Batch Mode (High Accuracy Option)
Goal: Improve precision on targeted video segments.
Mechanism: The user defines start_frame and end_frame to process only a specific chunk. This allows localized temporal context while keeping memory usage controlled.
Benefit: Offers a balance between high accuracy and manageable memory for complex or high-motion scenes.

