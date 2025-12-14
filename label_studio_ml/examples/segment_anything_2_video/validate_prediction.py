#!/usr/bin/env python3
"""
Quick validation script to test prediction format against Label Studio API.
Creates a minimal dummy prediction and attempts to upload it.
"""
import argparse
import json
import logging
import sys

from label_studio_sdk.client import LabelStudio

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_dummy_prediction(frames_count: int = 100, fps: float = 30.0) -> dict:
    """Create a minimal valid prediction to test the format."""
    duration = frames_count / fps if fps > 0 else 0.0
    return {
        "result": [
            {
                "id": "test-track-1",
                "type": "videorectangle",
                "from_name": "box",
                "to_name": "video",
                "score": 1.0,
                "origin": "manual",
                "value": {
                    "sequence": [
                        {"frame": 1, "x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0, "enabled": True, "rotation": 0, "time": 0.0},
                        {"frame": 50, "x": 15.0, "y": 15.0, "width": 20.0, "height": 20.0, "enabled": True, "rotation": 0, "time": 49.0 / fps},
                    ],
                    "framesCount": frames_count,
                    "duration": duration,
                    "labels": ["object"],
                },
                "meta": {"text": "id:"},
            }
        ],
        "score": 1.0,
        "model_version": "validation-test",
    }


def validate_prediction_format(prediction: dict) -> list:
    """Validate prediction format and return list of issues."""
    issues = []
    
    if "result" not in prediction:
        issues.append("Missing 'result' key")
        return issues

    # Top-level fields expected by uploader/model
    if "score" not in prediction:
        issues.append("Missing top-level 'score'")
    elif not isinstance(prediction.get("score"), (int, float)):
        issues.append(f"Top-level 'score' should be numeric, got {type(prediction.get('score'))}")
    if "model_version" not in prediction:
        issues.append("Missing top-level 'model_version'")
    elif not isinstance(prediction.get("model_version"), str):
        issues.append(f"Top-level 'model_version' should be a string, got {type(prediction.get('model_version'))}")
    
    for i, region in enumerate(prediction["result"]):
        prefix = f"Region {i}"
        
        # Required fields
        for field in ["id", "type", "from_name", "to_name", "value"]:
            if field not in region:
                issues.append(f"{prefix}: Missing required field '{field}'")

        if region.get("type") != "videorectangle":
            issues.append(f"{prefix}: type should be 'videorectangle', got '{region.get('type')}'")

        # Optional but expected fields from generators
        if "origin" not in region:
            issues.append(f"{prefix}: Missing 'origin'")
        if "score" in region and not isinstance(region.get("score"), (int, float)):
            issues.append(f"{prefix}: 'score' should be numeric, got {type(region.get('score'))}")
        if "meta" in region:
            meta = region.get("meta") or {}
            if "text" not in meta:
                issues.append(f"{prefix}: 'meta.text' missing")
            elif not isinstance(meta.get("text"), str):
                issues.append(f"{prefix}: 'meta.text' should be string, got {type(meta.get('text'))}")
        
        value = region.get("value", {})
        if "sequence" not in value:
            issues.append(f"{prefix}: Missing 'sequence' in value")
        if "framesCount" not in value:
            issues.append(f"{prefix}: Missing 'framesCount' in value")
        if "labels" not in value:
            issues.append(f"{prefix}: Missing 'labels' in value")
        elif not isinstance(value.get("labels"), list):
            issues.append(f"{prefix}: 'labels' should be a list, got {type(value.get('labels'))}")
        elif not value.get("labels"):
            issues.append(f"{prefix}: 'labels' list is empty")
        else:
            for lbl in value.get("labels", []):
                if not isinstance(lbl, str):
                    issues.append(f"{prefix}: label entries should be strings, got {type(lbl)}")

        if "duration" not in value:
            issues.append(f"{prefix}: Missing 'duration' in value")
        elif not isinstance(value.get("duration"), (int, float)):
            issues.append(f"{prefix}: 'duration' should be numeric, got {type(value.get('duration'))}")
        
        # Validate sequence items
        seq = value.get("sequence", [])
        for j, item in enumerate(seq):
            item_prefix = f"{prefix}, Frame {j}"
            for field in ["frame", "x", "y", "width", "height"]:
                if field in item and not isinstance(item[field], (int, float)):
                    issues.append(f"{item_prefix}: '{field}' should be numeric, got {type(item[field])}")
            if "frame" in item and not isinstance(item["frame"], int):
                issues.append(f"{item_prefix}: 'frame' should be int, got {type(item['frame'])}")
            if "enabled" in item and not isinstance(item.get("enabled"), bool):
                issues.append(f"{item_prefix}: 'enabled' should be bool, got {type(item.get('enabled'))}")
            if "rotation" in item and not isinstance(item.get("rotation"), (int, float)):
                issues.append(f"{item_prefix}: 'rotation' should be numeric, got {type(item.get('rotation'))}")
            if "time" in item:
                if not isinstance(item.get("time"), (int, float)):
                    issues.append(f"{item_prefix}: 'time' should be numeric, got {type(item.get('time'))}")
                elif item.get("time") < 0:
                    issues.append(f"{item_prefix}: 'time' should be non-negative")
    
    return issues


def test_upload(ls_url: str, api_key: str, task_id: int, dry_run: bool = True):
    """Test uploading a prediction to Label Studio."""
    ls = LabelStudio(base_url=ls_url, api_key=api_key)
    
    # Get task to find frames count
    task = ls.tasks.get(id=task_id)
    
    # Try to get frames count from task data
    frames_count = 100  # default
    if hasattr(task, "data") and isinstance(task.data, dict):
        # Look for video metadata
        for key, val in task.data.items():
            if isinstance(val, dict) and "framesCount" in val:
                frames_count = val["framesCount"]
                break
    
    logger.info(f"Task {task_id} - using framesCount={frames_count}")
    
    prediction = create_dummy_prediction(frames_count)
    
    # Validate format first
    issues = validate_prediction_format(prediction)
    if issues:
        logger.error("❌ Prediction format validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("✅ Prediction format validation passed")
    logger.info("\nPrediction JSON:")
    logger.info(json.dumps(prediction, indent=2))
    
    if dry_run:
        logger.info("\n[DRY RUN] Skipping actual upload")
        return True
    
    # Try actual upload
    logger.info("\nAttempting upload...")
    try:
        result = ls.predictions.create(
            task=task_id,
            score=prediction.get("score", 0.0),
            model_version=prediction.get("model_version", "test"),
            result=prediction.get("result", []),
        )
        pred_id = getattr(result, "id", None)
        logger.info(f"✅ Upload succeeded! Prediction ID: {pred_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate prediction format for Label Studio")
    parser.add_argument("--ls-url", required=True, help="Label Studio URL")
    parser.add_argument("--ls-api-key", required=True, help="Label Studio API key")
    parser.add_argument("--task", type=int, required=True, help="Task ID to test against")
    parser.add_argument("--upload", action="store_true", help="Actually upload (not just validate)")
    parser.add_argument("--prediction-file", type=str, help="Path to prediction JSON file to validate (optional)")
    
    args = parser.parse_args()
    
    if args.prediction_file:
        # Validate an existing prediction file
        with open(args.prediction_file) as f:
            prediction = json.load(f)
        
        issues = validate_prediction_format(prediction)
        if issues:
            logger.error("❌ Prediction format validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            sys.exit(1)
        else:
            logger.info("✅ Prediction format validation passed")
            
            if args.upload:
                ls = LabelStudio(base_url=args.ls_url, api_key=args.ls_api_key)
                try:
                    result = ls.predictions.create(
                        task=args.task,
                        score=prediction.get("score", 0.0),
                        model_version=prediction.get("model_version", "test"),
                        result=prediction.get("result", []),
                    )
                    logger.info(f"✅ Upload succeeded! Prediction ID: {getattr(result, 'id', 'unknown')}")
                except Exception as e:
                    logger.error(f"❌ Upload failed: {e}")
                    sys.exit(1)
    else:
        # Test with dummy prediction
        success = test_upload(args.ls_url, args.ls_api_key, args.task, dry_run=not args.upload)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
