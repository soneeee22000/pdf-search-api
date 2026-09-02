"""Recognition for pages that carry no text layer.

Only pages that produced no extractable text reach this module, which is what
keeps the cost bounded: on the supplied corpus that is 2 pages of 64. A scanned
page is otherwise invisible to the whole pipeline -- it yields no text, so no
chunks, so no vectors, and no embedding model can make it retrievable.

The engine is a seam with two implementations because which one to ship was
decided by measurement rather than preference; see `eval/OCR_DECISION.md`. Both
imports are deferred to construction so that neither is a hard dependency: a
build without an OCR engine installed degrades to the previous behaviour of
recording the page as empty and saying so, rather than failing to start.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# PDF user space is 72 units to the inch. Recognition accuracy on small print
# falls off below roughly 200 DPI, and rendering is cheap here because so few
# pages qualify, so the render is fixed at this resolution for every engine --
# holding it constant is what makes the comparison in OCR_DECISION.md a
# comparison of engines rather than of rasterisation settings.
OCR_RENDER_DPI = 216
PDF_UNITS_PER_INCH = 72
OCR_RENDER_SCALE = OCR_RENDER_DPI / PDF_UNITS_PER_INCH


class OcrUnavailableError(RuntimeError):
    """Raised when a named engine cannot be constructed."""


class OcrEngine(Protocol):
    """Turns a rendered page image into text."""

    @property
    def name(self) -> str:
        """Identifier recorded in the manifest."""
        ...

    def extract(self, image: NDArray[np.uint8]) -> str:
        """Return the recognised text, empty if nothing was found."""
        ...


class RapidOcrEngine:
    """ONNX detection and recognition, installed entirely through pip."""

    def __init__(self) -> None:
        """Load the ONNX models once, so the cost is not paid per page."""
        from rapidocr_onnxruntime import RapidOCR

        self._reader = RapidOCR()

    @property
    def name(self) -> str:
        """Identifier recorded in the manifest."""
        return "rapidocr"

    def extract(self, image: NDArray[np.uint8]) -> str:
        """Recognise text, joining the detected lines in reading order."""
        result, _elapsed = self._reader(image)
        if not result:
            return ""
        return "\n".join(str(line[1]) for line in result)


class TesseractEngine:
    """Tesseract through pytesseract, which needs the system binary."""

    LANGUAGE = "fra"

    def __init__(self) -> None:
        """Fail at construction if the binary is missing, not at first page."""
        import pytesseract

        self._pytesseract = pytesseract
        self._pytesseract.get_tesseract_version()

    @property
    def name(self) -> str:
        """Identifier recorded in the manifest."""
        return "tesseract"

    def extract(self, image: NDArray[np.uint8]) -> str:
        """Recognise text using the French language data."""
        return str(self._pytesseract.image_to_string(image, lang=self.LANGUAGE))


_ENGINES: dict[str, type] = {
    "rapidocr": RapidOcrEngine,
    "tesseract": TesseractEngine,
}

DISABLED = ("", "none", "off")


def available_engines() -> tuple[str, ...]:
    """Names accepted by `build_engine`, for the CLI's help text."""
    return tuple(_ENGINES)


def build_engine(name: str | None) -> OcrEngine | None:
    """Construct an engine by name, or None when OCR is switched off.

    An unknown name is an error rather than a silent fallback: a typo that
    quietly disabled OCR would show up only as a document missing from the
    index, which is the failure this module exists to remove.
    """
    key = (name or "").strip().lower()
    if key in DISABLED:
        return None
    if key not in _ENGINES:
        raise OcrUnavailableError(
            f"unknown OCR engine {name!r}; available: {', '.join(available_engines())}"
        )
    try:
        engine: OcrEngine = _ENGINES[key]()
    except Exception as exc:  # noqa: BLE001 - missing package or missing binary
        raise OcrUnavailableError(
            f"OCR engine {key!r} is named but not usable here: {exc}. "
            "Install it, or run without --ocr to index only pages that have a text layer."
        ) from exc
    logger.info("OCR enabled: %s at %d DPI", key, OCR_RENDER_DPI)
    return engine
