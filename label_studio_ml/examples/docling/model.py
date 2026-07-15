"""Docling SaaS (IBM) via DoclingServiceClient -> Label Studio predictions.

Predictions target the **HumanSignal Interfaces** Docling annotator
(``docling-ls-implementation/docling_interface.jsx``), which reads results
through ``parseResults``. Output is canonical Label Studio result shapes —
``rectanglelabels`` and ``polygonlabels`` — built by
:mod:`docling_to_ls_results`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import (
    get_local_path as ls_sdk_get_local_path,
)

from docling.datamodel.base_models import ConversionStatus
from docling.service_client import DoclingServiceClient
from docling.service_client.exceptions import ConversionError, DoclingServiceClientError

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_image_size
from label_studio_sdk.label_interface.objects import PredictionValue

from docling_to_ls_results import docling_document_to_ls_results, page_raster_size

logger = logging.getLogger(__name__)

# Avoid repeating the same configuration warning on every prediction request.
_PLACEHOLDER_URL_WARNED = False


def _docling_service_client_base_url(raw: str) -> str:
    """IBM SaaS URLs look like ``…/<tenant>/v1``; ``DoclingServiceClient`` still prefixes routes with ``/v1``
    (yielding ``…/v1/v1/…`` and 404/400). Strip one trailing ``/v1`` so the client gets ``…/<tenant>``.
    """
    base = raw.strip().rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")].rstrip("/")
    return base


class Docling(LabelStudioMLBase):
    """Use Docling SaaS through ``DoclingServiceClient.convert`` (URL or local ``Path``)."""

    # Model state and temporary downloads live under MODEL_DIR so Docker volumes can persist cache files
    # without polluting the example directory.
    MODEL_DIR = os.getenv("MODEL_DIR", ".")

    # Workbench provides a tenant-specific URL, typically ending in /v1. We keep the raw value here
    # and normalize only when constructing DoclingServiceClient.
    DOCLING_SERVICE_URL = (
        os.getenv("DOCLING_SERVICE_URL", "").strip().rstrip("/")
        or os.getenv("DOCLING_SERVE_URL", "").strip().rstrip("/")
    )
    DOCLING_API_KEY = os.getenv("DOCLING_SERVE_API_KEY") or os.getenv("DOCLING_API_KEY") or ""

    _client: Optional[DoclingServiceClient] = None

    def __init__(self, project_id: Optional[str] = None, label_config: Optional[str] = None, **kwargs):
        # Optional overrides. The Docling Interface (docling_interface.jsx) hardcodes
        # from_name="docling" / to_name="docling" and reads task.data.image, so the
        # defaults below match — set these env vars only if your project overrides
        # those names.
        # DOCLING_REACTCODE_FROM_NAME / _TO_NAME are the legacy env var names kept
        # for backward compatibility; DOCLING_FROM_NAME / _TO_NAME are the preferred
        # names going forward.
        self._from_name = (
            os.getenv("DOCLING_FROM_NAME")
            or os.getenv("DOCLING_REACTCODE_FROM_NAME")
            or "docling"
        )
        self._to_name = (
            os.getenv("DOCLING_TO_NAME")
            or os.getenv("DOCLING_REACTCODE_TO_NAME")
            or "docling"
        )
        self._data_key = os.getenv("DOCLING_TASK_DATA_KEY") or "image"

        # HumanSignal Interfaces projects ship a near-empty ``<View></View>`` label_config
        # and load the actual interface from ``custom_interface_code`` (which the ML
        # backend never sees). We still validate opportunistically so the SDK schema
        # loads when a project happens to ship a parseable one, but a parse failure
        # is not fatal — the backend has all it needs from env vars and defaults.
        label_config_for_sdk = label_config
        if label_config:
            try:
                from label_studio_sdk.label_interface import LabelInterface

                LabelInterface(config=label_config)
            except Exception as exc:
                logger.warning(
                    "LabelInterface could not parse label_config (%s); continuing without SDK schema",
                    exc,
                )
                label_config_for_sdk = None
        super().__init__(project_id=project_id, label_config=label_config_for_sdk, **kwargs)

    def setup(self) -> None:
        # Expose the installed docling package version in predictions so users can trace which client
        # library produced a result.
        try:
            import docling as dl

            ver = getattr(dl, "__version__", "unknown")
        except Exception:
            ver = "unknown"
        self.set("model_version", f"DoclingService-{ver}")

    def _ensure_client(self) -> DoclingServiceClient:
        # DoclingServiceClient keeps HTTP clients/watchers open, so reuse one client per process
        # instead of recreating it for every task.
        if self._client is not None:
            return self._client
        if not self.DOCLING_SERVICE_URL:
            raise ValueError(
                "Set DOCLING_SERVICE_URL to your SaaS base URL from Workbench "
                "(typically ending in /v1), e.g. https://api.aws-c1.dcls.saas.ibm.com/<instance>/v1"
            )
        client_url = _docling_service_client_base_url(self.DOCLING_SERVICE_URL)
        if client_url != self.DOCLING_SERVICE_URL.strip().rstrip("/"):
            logger.info(
                "Using DoclingServiceClient base URL without trailing /v1 (client adds /v1 on routes): %s",
                client_url,
            )
        timeout = float(os.getenv("DOCLING_SERVE_TIMEOUT", "600"))
        self._client = DoclingServiceClient(
            url=client_url,
            api_key=self.DOCLING_API_KEY,
            job_timeout=timeout,
            http_read_timeout=timeout,
            http_connect_timeout=float(os.getenv("DOCLING_HTTP_CONNECT_TIMEOUT", "30")),
        ).__enter__()
        return self._client

    def _convert_with_service(
        self,
        client: DoclingServiceClient,
        *,
        source: Path | str,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Call SaaS ``convert``; raises ConversionError on failure when raises_on_error=True."""
        extra = os.getenv("DOCLING_CONVERT_SOURCE_HEADERS_JSON")
        if extra:
            merged = json.loads(extra)
            headers = {**(headers or {}), **merged}
        return client.convert(source=source, headers=headers or None)

    def _task_file_url(self, task: Dict[str, Any]) -> Optional[str]:
        # Match docling_interface.jsx's resolveImageUrl fallback chain so the ML backend reads from
        # the same place the labeling iframe does — including LSE's ``$undefined$`` direct-upload key
        # (lowercase, with a leading and trailing dollar sign).
        data = task.get("data") or {}
        # DATA_UNDEFINED_NAME is "$undefined$"; keep it in the chain rather than as a separate
        # fallback below, so it goes through the same dict handling as every other key.
        keys_to_try: List[Optional[str]] = [
            self._data_key,
            "image",
            "url",
            "ocr",
            DATA_UNDEFINED_NAME,
            "$undefined",
            "undefined",
            "pdf",
            "document",
            "file",
        ]
        for key in keys_to_try:
            if not key:
                continue
            val = data.get(key)
            if val:
                # Mirrors the interface's "object with a .url field" affordance.
                if isinstance(val, dict):
                    inner = val.get("url") or val.get("URL")
                    if inner:
                        return str(inner)
                    continue
                return str(val)
        # Last resort: scan for values that look like files. This keeps the backend usable even when
        # the task data key and labeling config do not agree.
        for _k, val in data.items():
            if isinstance(val, str) and (
                val.startswith("http://")
                or val.startswith("https://")
                or val.startswith("/data/")
                or val.startswith("/storage-data/")
                or val.startswith("s3://")
            ):
                return val
        return None

    @staticmethod
    def _label_studio_hostname_for_download() -> Optional[str]:
        """Base URL of Label Studio for resolving upload/storage paths (read at request time, not import time)."""
        raw = (os.getenv("LABEL_STUDIO_URL") or os.getenv("LABEL_STUDIO_HOST") or "").strip()
        if not raw:
            return None
        return raw.rstrip("/")

    @staticmethod
    def _needs_label_studio_download(url: str) -> bool:
        # Relative LS upload/storage URLs and cloud-storage URIs require Label Studio to resolve or
        # presign the file. Public HTTP(S) URLs can be downloaded directly or sent to Docling SaaS.
        u = url.strip()
        if u.startswith(("http://", "https://")):
            return False
        return u.startswith("/") or u.startswith("s3://")

    def predict_single(self, task: Dict[str, Any]) -> Optional[PredictionValue]:
        """
        This method is called for each task in the batch.
        It is used to predict the regions of the document.
        """
        url_raw = self._task_file_url(task)
        url = (url_raw or "").strip()
        if not url:
            logger.warning(
                "No file URL found in task %s data keys=%s",
                task.get("id"),
                list((task.get("data") or {}).keys()),
            )
            return None

        use_remote_url = os.getenv("DOCLING_CONVERT_REMOTE_URL_ONLY", "").lower() in (
            "1",
            "true",
            "yes",
        )
        path: Optional[Path] = None
        if not use_remote_url or not url.lower().startswith(("http://", "https://")):
            # Default to downloading through Label Studio first. This handles private uploads,
            # storage proxy URLs, and cloud storage integrations that Docling SaaS cannot fetch directly.
            ls_hostname = self._label_studio_hostname_for_download()
            needs_ls = self._needs_label_studio_download(url)
            if needs_ls and not ls_hostname:
                logger.error(
                    "Task %s file URL %s requires Label Studio to download the file. "
                    "Set LABEL_STUDIO_URL (preferred) or LABEL_STUDIO_HOST to your Label Studio base URL "
                    "(with http:// or https://), reachable from this container, "
                    "and LABEL_STUDIO_API_KEY or LABEL_STUDIO_ACCESS_TOKEN.",
                    task.get("id"),
                    url[:160] + ("..." if len(url) > 160 else ""),
                )
                return None
            token = self.get_label_studio_access_token()
            if needs_ls and not token:
                logger.error(
                    "Task %s file URL %s requires LABEL_STUDIO_API_KEY or LABEL_STUDIO_ACCESS_TOKEN "
                    "so the backend can download uploads/storage-proxy URLs from Label Studio.",
                    task.get("id"),
                    url[:160] + ("..." if len(url) > 160 else ""),
                )
                return None
            cache_dir = os.path.join(self.MODEL_DIR, ".file-cache")
            os.makedirs(cache_dir, exist_ok=True)
            # Call the SDK directly so hostname, token, and cache_dir are always applied. Some
            # label-studio-ml wrapper versions wire get_local_path kwargs inconsistently.
            try:
                local = ls_sdk_get_local_path(
                    url,
                    cache_dir=cache_dir,
                    hostname=ls_hostname,
                    access_token=token,
                    task_id=task.get("id"),
                    download_resources=True,
                )
            except FileNotFoundError as exc:
                logger.error(
                    "Could not resolve or download file for task %s. "
                    "If the URL is under /storage-data/ or /data/, set LABEL_STUDIO_URL to the exact origin "
                    "you use in the browser (scheme + host + port), ensure the ML container can reach it "
                    "(not localhost from inside Docker unless using host.docker.internal), and set "
                    "LABEL_STUDIO_API_KEY. Original error: %s",
                    task.get("id"),
                    exc,
                )
                return None
            except ValueError as exc:
                logger.error(
                    "Invalid Label Studio base URL (must start with http:// or https://): %s",
                    exc,
                )
                return None
            except requests.exceptions.RequestException as exc:
                resp = getattr(exc, "response", None)
                detail = ""
                if resp is not None:
                    detail = f" HTTP {resp.status_code}"
                    try:
                        detail += f" body[:200]={resp.text[:200]!r}"
                    except Exception:
                        pass
                logger.error(
                    "Download failed for task %s URL %s.%s — check LABEL_STUDIO_URL, API key, TLS "
                    "(try VERIFY_SSL=false for self-signed), and network from this container to Label Studio. "
                    "Underlying error: %s",
                    task.get("id"),
                    url[:160],
                    detail,
                    exc,
                )
                return None
            path = Path(local)

        # Log source and resolved file size before conversion; this is the fastest way to spot bad
        # Label Studio URL/token settings (missing or tiny downloads).
        try:
            sz = path.stat().st_size if path else -1
        except OSError:
            sz = -1
        logger.info(
            "Docling task %s: path=%s size=%s source=%s remote_only=%s",
            task.get("id"),
            path or url,
            sz,
            url[:120] + ("..." if len(url) > 120 else ""),
            use_remote_url,
        )

        try:
            client = self._ensure_client()
        except ValueError as exc:
            logger.error("%s", exc)
            return None

        # When remote-only mode is enabled, pass public HTTP(S) URLs straight to SaaS just like the
        # Workbench snippet. Otherwise send the cached local Path that we downloaded from Label Studio.
        convert_source: Path | str = url if use_remote_url and url.lower().startswith(("http://", "https://")) else (path if path is not None else url)

        try:
            result = self._convert_with_service(client, source=convert_source)
        except ConversionError as exc:
            logger.error("Docling conversion failed for task %s: %s", task.get("id"), exc)
            return None
        except DoclingServiceClientError as exc:
            logger.error("Docling client error for task %s: %s", task.get("id"), exc)
            return None
        except Exception:
            logger.exception("Docling convert raised for task %s", task.get("id"))
            return None

        # Treat partial success as usable: Docling may still return a document with page-level issues
        # that can be mapped into Label Studio predictions.
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            logger.error(
                "Docling conversion status for task %s: %s errors=%s",
                task.get("id"),
                result.status,
                getattr(result, "errors", None),
            )
            return None

        doc = result.document
        if doc is None:
            logger.error("Docling returned no document for task %s", task.get("id"))
            return None

        # Optional conversion filters/toggles let users reduce output volume or include reading-order
        # metadata without changing code.
        page_raw = os.getenv("DOCLING_PAGE_NO", "").strip()
        page_no: Optional[int] = int(page_raw) if page_raw.isdigit() else None

        include_ro = os.getenv("DOCLING_PREDICT_READING_ORDER", "").lower() in ("1", "true", "yes")
        ro_level = int(os.getenv("DOCLING_READING_ORDER_LEVEL", "1") or "1")
        content_layers = os.getenv("DOCLING_CONTENT_LAYERS")

        # original_width / original_height must describe the raster the percentages were
        # measured against, because LS-native consumers multiply the two back together to
        # recover pixels. Docling's own page raster is that source and is always available —
        # unlike get_image_size, which cannot open a PDF (the primary input for this backend).
        iw = ih = 0
        raster = page_raster_size(doc, page_no)
        if raster:
            iw, ih = raster
        elif path is not None:
            # No page raster (unusual): fall back to probing the downloaded file, which only
            # works when the task file is itself an image.
            try:
                iw, ih = get_image_size(str(path))
            except Exception as exc:
                logger.warning(
                    "Could not read image size for task %s from %s: %s", task.get("id"), path, exc
                )
        if not iw or not ih:
            iw, ih = 100, 100
            logger.warning(
                "Task %s: no page raster or image size available; emitting placeholder "
                "original_width/original_height=%sx%s. Percent coordinates stay correct, but "
                "consumers converting them back to pixels will be wrong.",
                task.get("id"),
                iw,
                ih,
            )

        # Canonical Label Studio result shapes (rectanglelabels / polygonlabels),
        # matching docling_interface.jsx's parseResults contract.
        canonical_results = docling_document_to_ls_results(
            doc,
            page_no=page_no,
            include_reading_order=include_ro,
            reading_order_level=ro_level,
            content_layers=content_layers,
            from_name=self._from_name,
            to_name=self._to_name,
        )
        ls_results: List[Dict[str, Any]] = []
        for entry in canonical_results:
            # Each canonical entry already carries id / from_name / to_name / type / value /
            # origin. The ML backend only needs to bolt on the image dimensions.
            ls_results.append(
                {
                    **entry,
                    "original_width": iw,
                    "original_height": ih,
                    "image_rotation": 0,
                }
            )
        region_count = len(canonical_results)

        # Use a simple confidence proxy so non-empty predictions sort above empty ones without implying
        # calibrated model probabilities.
        score = min(1.0, 0.5 + 0.01 * region_count) if region_count else 0.0
        return PredictionValue(result=ls_results, score=score)

    def predict(self, tasks: List[Dict[str, Any]], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        global _PLACEHOLDER_URL_WARNED

        # Log the batch shape up front; Label Studio often shows only a generic "no response" when
        # the backend returns an empty prediction list.
        tasks = tasks or []
        logger.info(
            "Docling predict: %s task(s), ids=%s data_keys=%s",
            len(tasks),
            [t.get("id") for t in tasks],
            [list((t.get("data") or {}).keys()) for t in tasks],
        )

        svc = self.DOCLING_SERVICE_URL
        # The compose template intentionally ships with a placeholder; warn once so repeated predict
        # requests do not flood logs.
        if svc and "YOUR_INSTANCE" in svc and not _PLACEHOLDER_URL_WARNED:
            logger.error(
                "DOCLING_SERVICE_URL still contains a placeholder — set the full SaaS URL from Workbench."
            )
            _PLACEHOLDER_URL_WARNED = True

        if not svc:
            logger.error("DOCLING_SERVICE_URL is empty — set it in the environment (see README).")

        predictions = []
        for task in tasks:
            # Process tasks independently so one bad file or conversion does not fail the whole batch.
            prediction = self.predict_single(task)
            if prediction:
                predictions.append(prediction)

        # Empty predictions are valid HTTP responses, but usually mean configuration or file resolution
        # failed; make logs actionable for users debugging from Docker output.
        if tasks and not predictions:
            logger.warning(
                "Docling produced zero predictions for %s task(s). "
                "Check ERROR logs above; ensure LABEL_STUDIO_URL/API_KEY for uploads; "
                "or try DOCLING_CONVERT_REMOTE_URL_ONLY=true with a public HTTPS URL.",
                len(tasks),
            )
        elif predictions:
            logger.info("Docling predict finished: %s non-empty result(s)", len(predictions))

        return ModelResponse(predictions=predictions, model_version=str(self.get("model_version") or ""))

    def fit(self, event, data, **kwargs):
        return {}
