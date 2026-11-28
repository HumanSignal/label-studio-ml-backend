"""Label Studio ML backend for ReDefiNER: pre-annotates spans using the project's labels and their descriptions as a
plain-English rulebook.

The rulebook comes from, in order:
1. RULEBOOK=default: the model page's own rulebook (pages such as jungroo-ai/redefiner-pii ship one).
2. RULEBOOK set to rulebook text, a rulebook file, or a folder holding <project id>.txt files.
3. Otherwise the labeling config: each <Label value="X" hint="..."/> becomes [TYPE] X with the hint as its description.

ReDefiNER is not fine-tuned on annotations. To fix a systematic mistake, add a rule line to the rulebook.
"""
import hashlib
import logging
import os
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("REDEFINER_MODEL", "jungroo-ai/redefiner-0.8b")
GENERAL_RULE = "Extract every entity of the supplied types. Return exact spans from the text."
TYPE_LINE = re.compile(r"^\[TYPE\]\s*(.+?)\s*$", re.MULTILINE)

_taggers = {}
_lock = threading.Lock()


def get_tagger():
    """One loaded model per process: Label Studio creates a new backend object for every request."""
    with _lock:
        if MODEL_NAME not in _taggers:
            from redefiner import Tagger
            logger.info(f"Loading ReDefiNER model {MODEL_NAME}")
            kwargs = {}
            if os.getenv("MAX_SPAN_TOKENS"):
                kwargs["max_span_tokens"] = int(os.getenv("MAX_SPAN_TOKENS"))
            _taggers[MODEL_NAME] = Tagger.from_pretrained(MODEL_NAME, device=os.getenv("DEVICE") or None, **kwargs)
        return _taggers[MODEL_NAME]


def rulebook_from_labels(label_config: str, from_name: str) -> str:
    """A rulebook with one [TYPE] per <Label> of the <Labels> tag `from_name`, its `hint` as the description."""
    from redefiner import to_text

    root = ET.fromstring(label_config)
    labels_tag = next((t for t in root.iter("Labels") if t.get("name") == from_name), None)
    if labels_tag is None:
        raise ValueError(f'no <Labels name="{from_name}"> in the labeling config')
    classes = []
    for label in labels_tag.iter("Label"):
        value = label.get("value")
        if not value:
            continue
        cls = {"name": value}
        hint = " ".join((label.get("hint") or "").split())
        if hint:
            cls["description"] = hint
        classes.append(cls)
    if not classes:
        raise ValueError(f'<Labels name="{from_name}"> has no <Label value="..."/> entries')
    return to_text({"general": [GENERAL_RULE], "classes": classes})


def rulebook_from_setting(setting: str, project_id: str) -> Optional[str]:
    """The RULEBOOK setting as rulebook text: the text itself, a file, or <folder>/<project id>.txt.

    Returns None when RULEBOOK is a folder without a file for this project (the labels are used instead).
    """
    if "[TYPE]" in setting:
        return setting
    path = Path(setting)
    if path.is_dir():
        project_file = path / f"{str(project_id).split('.')[0]}.txt"
        return project_file.read_text(encoding="utf-8") if project_file.is_file() else None
    if path.is_file():
        return path.read_text(encoding="utf-8")
    raise ValueError(f"RULEBOOK={setting!r} is not rulebook text (no [TYPE] line), a file, a folder or 'default'")


class ReDefiNERModel(LabelStudioMLBase):
    """Pre-annotation with ReDefiNER: entity spans defined by a plain-English rulebook."""

    def setup(self):
        self.threshold = float(os.getenv("THRESHOLD")) if os.getenv("THRESHOLD") else None
        self.batch_size = int(os.getenv("BATCH_SIZE", 16))

    def rulebook(self, from_name: str) -> Optional[str]:
        """The rulebook for this project; None means the model page's default rulebook."""
        setting = os.getenv("RULEBOOK", "").strip()
        if setting == "default":
            return None
        if setting:
            text = rulebook_from_setting(setting, self.project_id)
            if text is not None:
                return text
        return rulebook_from_labels(self.label_config, from_name)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        from_name, to_name, value = self.label_interface.get_first_tag_occurence("Labels", "Text")
        labels = set(self.label_interface.get_tag(from_name).labels)

        rulebook = self.rulebook(from_name)
        tagger = get_tagger()
        if rulebook is None and getattr(tagger, "default_rulebook", None) is None:
            raise ValueError(f"RULEBOOK=default, but {MODEL_NAME} has no default rulebook")

        types = set(TYPE_LINE.findall(rulebook if rulebook is not None else tagger.default_rulebook))
        missing = sorted(labels - types)
        if missing:
            logger.warning(f"Labels with no [TYPE] in the rulebook, never predicted: {missing}")

        # a new rulebook gives a new model version, so Label Studio keeps each rulebook's predictions apart
        digest = hashlib.sha256(f"{MODEL_NAME}\n{rulebook or 'default'}".encode()).hexdigest()[:8]
        model_version = f"redefiner-{digest}"

        texts = [str(task["data"].get(value) or "") for task in tasks]
        found = tagger.extract(texts, rulebook, threshold=self.threshold, batch_size=self.batch_size)

        predictions = []
        for spans in found:
            result = [{
                "from_name": from_name,
                "to_name": to_name,
                "type": "labels",
                "value": {"start": s["start"], "end": s["end"], "text": s["text"], "labels": [s["label"]]},
                "score": round(float(s["confidence"]), 4),
            } for s in spans if s["label"] in labels]   # drops NOT_AN_ENTITY and types the project does not use
            # the least confident span ranks the task for review; a task with no spans gets no score
            score = min(r["score"] for r in result) if result else None
            predictions.append(PredictionValue(result=result, score=score, model_version=model_version))

        self.set("model_version", model_version)
        return ModelResponse(predictions=predictions, model_version=model_version)
