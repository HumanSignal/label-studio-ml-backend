"""Unit-level tests for ``docling_to_ls_results``.

These tests use lightweight stubs in place of the real ``DoclingDocument`` so
they run without IBM Docling SaaS access. They assert the *result envelope
shape* only — that is exactly the contract that
``docling-ls-implementation/docling_interface.jsx`` ``parseResults`` reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel

from docling_to_ls_results import docling_document_to_ls_results


class _FakeBBox:
    def __init__(self, l: float, t: float, r: float, b: float):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.width = r - l
        self.height = b - t

    def to_top_left_origin(self, page_height: float) -> "_FakeBBox":
        # Pretend our bbox is already top-left; production code calls this regardless.
        return self

    def scale_to_size(self, *, old_size, new_size) -> "_FakeBBox":
        # Scale unchanged — keep coordinates simple for the test.
        sx = new_size.width / old_size.width
        sy = new_size.height / old_size.height
        return _FakeBBox(self.l * sx, self.t * sy, self.r * sx, self.b * sy)


def _fake_size(w: float, h: float) -> SimpleNamespace:
    return SimpleNamespace(width=w, height=h)


def _fake_page(size_w: float = 100.0, size_h: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(size=_fake_size(size_w, size_h), image=None)


def _fake_item(
    *,
    label: DocItemLabel,
    bbox: _FakeBBox,
    page_no: int = 1,
    text: str = "",
    content_layer: ContentLayer = ContentLayer.BODY,
) -> SimpleNamespace:
    prov = SimpleNamespace(page_no=page_no, bbox=bbox)
    return SimpleNamespace(
        prov=[prov],
        label=label,
        text=text,
        content_layer=content_layer,
        meta=None,
    )


class _FakeDoc:
    def __init__(self, items_with_levels: List[Tuple[Any, int]]):
        self._items = items_with_levels
        self.pages: Dict[int, Any] = {1: _fake_page(100.0, 100.0)}

    def iterate_items(self, **_kwargs):
        for item, level in self._items:
            yield item, level


def test_rectanglelabels_envelope_shape() -> None:
    item = _fake_item(
        label=DocItemLabel.SECTION_HEADER,
        bbox=_FakeBBox(10, 20, 30, 40),
        text="Hello",
    )
    doc = _FakeDoc([(item, 1)])

    out = docling_document_to_ls_results(doc, from_name="docling", to_name="docling")
    assert len(out) == 1
    r = out[0]
    assert r["type"] == "rectanglelabels"
    assert r["from_name"] == "docling"
    assert r["to_name"] == "docling"
    assert r["origin"] == "prediction"
    assert "id" in r and isinstance(r["id"], str)

    v = r["value"]
    # Coordinates are in percent of the page raster.
    assert v["x"] == 10.0
    assert v["y"] == 20.0
    assert v["width"] == 20.0
    assert v["height"] == 20.0
    assert v["rotation"] == 0
    assert v["rectanglelabels"] == ["section_header"]
    assert v["content_layer"] == "BODY"
    assert v["level"] == 1
    assert v["picture_type"] is None
    assert v["text"] == "Hello"
    assert v["parentId"] is None


def test_reading_order_polyline_envelope_shape() -> None:
    items = [
        (
            _fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(0, 0, 10, 10), text="a"),
            1,
        ),
        (
            _fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(40, 40, 60, 60), text="b"),
            1,
        ),
    ]
    doc = _FakeDoc(items)

    out = docling_document_to_ls_results(
        doc, include_reading_order=True, reading_order_level=1
    )
    # Two rectangles + one polyline.
    types = [r["type"] for r in out]
    assert types.count("rectanglelabels") == 2
    assert types.count("polygonlabels") == 1

    poly = next(r for r in out if r["type"] == "polygonlabels")
    v = poly["value"]
    assert v["polygonlabels"] == ["reading_order"]
    assert v["closed"] is False
    assert isinstance(v["points"], list) and len(v["points"]) == 2
    # connectedRegions references the rectangle ids.
    rect_ids = [r["id"] for r in out if r["type"] == "rectanglelabels"]
    assert v["connectedRegions"] == rect_ids
    assert v["level"] == 1
    assert v["validationErrors"] == []
    assert v["parentId"] is None


def test_no_underscore_prefixed_keys_in_value() -> None:
    """The interface's spatial-region serialization validator rejects
    ``value`` payloads that leak in-memory underscore-prefixed keys."""
    item = _fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(0, 0, 10, 10))
    doc = _FakeDoc([(item, 1)])
    out = docling_document_to_ls_results(doc)
    assert out, "expected at least one result"
    for r in out:
        for k in (r.get("value") or {}).keys():
            assert not k.startswith("_"), f"underscore-prefixed key {k!r} leaked into value"


# ---------------------------------------------------------------------------
# Result-format auto-detection.
# ---------------------------------------------------------------------------


def _detect(label_config):
    # Imported lazily so the module under test doesn't pull in label_studio_ml
    # at collection time when only the converter tests are exercised. We
    # import the standalone module-level function, not the Docling class
    # itself, to keep this test free of label_studio_ml / docling deps.
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "_docling_model_for_test",
        pathlib.Path(__file__).parent / "model.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load model.py for test")
    # Stub out heavy deps before exec so import errors don't bubble up; only
    # ``detect_result_format`` is exercised here and it has no runtime deps.
    import sys
    import types

    for mod_name in (
        "label_studio_sdk",
        "label_studio_sdk._extensions",
        "label_studio_sdk._extensions.label_studio_tools",
        "label_studio_sdk._extensions.label_studio_tools.core",
        "label_studio_sdk._extensions.label_studio_tools.core.utils",
        "label_studio_sdk._extensions.label_studio_tools.core.utils.io",
        "label_studio_sdk.label_interface",
        "label_studio_sdk.label_interface.objects",
        "docling",
        "docling.datamodel",
        "docling.datamodel.base_models",
        "docling.service_client",
        "docling.service_client.exceptions",
        "label_studio_ml",
        "label_studio_ml.model",
        "label_studio_ml.response",
        "label_studio_ml.utils",
    ):
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    # Minimal attrs needed by the from-imports in model.py
    sys.modules["label_studio_sdk._extensions.label_studio_tools.core.utils.io"].get_local_path = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["docling.datamodel.base_models"].ConversionStatus = type("CS", (), {"SUCCESS": "S", "PARTIAL_SUCCESS": "P"})  # type: ignore[attr-defined]
    sys.modules["docling.service_client"].DoclingServiceClient = type("DSC", (), {})  # type: ignore[attr-defined]
    sys.modules["docling.service_client.exceptions"].ConversionError = type("CE", (Exception,), {})  # type: ignore[attr-defined]
    sys.modules["docling.service_client.exceptions"].DoclingServiceClientError = type("DCE", (Exception,), {})  # type: ignore[attr-defined]
    sys.modules["label_studio_ml.model"].LabelStudioMLBase = type("Base", (), {"__init__": lambda self, **k: None, "set": lambda self, *a, **k: None, "get": lambda self, *a, **k: None})  # type: ignore[attr-defined]
    sys.modules["label_studio_ml.response"].ModelResponse = type("MR", (), {})  # type: ignore[attr-defined]
    sys.modules["label_studio_ml.utils"].DATA_UNDEFINED_NAME = "$undefined$"  # type: ignore[attr-defined]
    sys.modules["label_studio_ml.utils"].get_image_size = lambda *a, **k: (100, 100)  # type: ignore[attr-defined]
    sys.modules["label_studio_sdk.label_interface.objects"].PredictionValue = type("PV", (), {})  # type: ignore[attr-defined]

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.detect_result_format(label_config)


def test_detect_reactcode_legacy_config() -> None:
    cfg = '<View><Image name="img" value="$image"/><ReactCode name="docling" toName="img"/></View>'
    assert _detect(cfg) == "reactcode"


def test_detect_canonical_when_no_reactcode_tag() -> None:
    # New HumanSignal Interface projects ship a near-empty View; the real
    # interface code lives in custom_interface_code which the ML backend
    # can't see. Absence of <ReactCode> is the reliable signal.
    assert _detect("<View></View>") == "canonical"


def test_detect_canonical_when_no_label_config() -> None:
    assert _detect(None) == "canonical"
    assert _detect("") == "canonical"


def test_detect_case_insensitive() -> None:
    assert _detect('<view><reactcode name="x" toname="y"/></view>') == "reactcode"
