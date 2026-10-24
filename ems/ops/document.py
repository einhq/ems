"""`document.extract` — the text of a PDF, DOCX, ODT, picture or plain-text file.

A scanned PDF has no text layer, so `get_text()` returns "" for every page and the document
would be indexed as empty — no error, simply never findable. `ocr: auto` rasterizes any page
that yields (almost) nothing and reads it with the text reader, the model family `ocr` behind
`ocr.read` (PP-OCRv6); `always` reads every page; `never` reads only the text layer. A picture is
read whole by the same reader, in the display frame the file reference names.

The reader tells the language of each picture or page itself (`ocr.read`), so `languages` is accepted
and changes nothing.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Literal

from fastapi import APIRouter, Request

from ..context import Context
from ..errors import EmsError, TOOL_FAILED, UNSUPPORTED
from ..schemas import FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ._models import call_on_pictures, mimetype_of, model_slot
from .media import kind_of

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
ODT = "application/vnd.oasis.opendocument.text"

OCR_DPI = int(os.environ.get("EMS_OCR_DPI", "200"))
MIN_PAGE_TEXT_CHARS = int(os.environ.get("EMS_OCR_MIN_PAGE_CHARS", "32"))
MAX_OCR_PAGES = int(os.environ.get("EMS_OCR_MAX_PAGES", "40"))

# Failures that are a property of the file, not of the run. Named by module.class so this
# costs no imports; deliberately narrow — anything unrecognised stays transient and loud.
PERMANENT_ERRORS = frozenset([
    "PIL.UnidentifiedImageError",
    "pymupdf.FileDataError",
    "pymupdf.EmptyFileError",
    "zipfile.BadZipFile",
])


class DocumentRequest(WithOptions):
    file: FileRef
    ocr: Literal["auto", "always", "never"] = "auto"
    languages: list[str] | None = None


def is_permanent(exc: Exception) -> bool:
    return f"{type(exc).__module__}.{type(exc).__name__}" in PERMANENT_ERRORS


def _pixmap_png(page) -> bytes:
    import fitz

    return page.get_pixmap(matrix=fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)).tobytes("png")


class Reader:
    """The text reader for one request: reads a picture's lines and keeps the engine that read them."""

    def __init__(self, ctx: Context, req: "DocumentRequest"):
        self.ctx = ctx
        self.req = req
        self.model: str | None = None

    async def read(self, path: str, mimetype: str | None, angle: int, mirror: bool) -> str:
        async with model_slot(self.ctx, self.req.options):
            raw = await call_on_pictures(self.ctx, "read_text", [path], angle, mirror, mimetype=mimetype, deadline=self.req.options.deadline)
        self.model = str(raw.get("model") or "ocr")
        return str(raw.get("text") or "").strip()


async def _ocr_page(page, reader: Reader, scratch: str) -> str:
    """A page rasterized and read. A page the reader cannot open has no text; a reader that is
    busy or down fails the request, so the document is not stored as empty."""
    try:
        png = await asyncio.to_thread(_pixmap_png, page)
    except Exception:  # noqa: BLE001 - a page that will not render has no text to read
        return ""
    source = os.path.join(scratch, f"page-{page.number}.png")
    with open(source, "wb") as handle:
        handle.write(png)
    try:
        return await reader.read(source, "image/png", 0, False)
    except EmsError as exc:
        if exc.permanent:
            return ""
        raise
    finally:
        try:
            os.unlink(source)
        except OSError:
            pass


async def extract_pdf(path: str, ocr: str, reader: Reader, scratch: str) -> tuple[str, int, str]:
    import fitz

    doc = await asyncio.to_thread(fitz.open, path)
    pages = len(doc)
    parts: list[str] = []
    ocr_pages = 0
    try:
        for page in doc:
            text = (await asyncio.to_thread(page.get_text)).strip() if ocr != "always" else ""
            wants_ocr = ocr == "always" or (ocr == "auto" and len(text) < MIN_PAGE_TEXT_CHARS)
            if wants_ocr and ocr_pages < MAX_OCR_PAGES:
                recognised = await _ocr_page(page, reader, scratch)
                if len(recognised) > len(text):
                    text = recognised
                    ocr_pages += 1
            if text:
                parts.append(text)
    finally:
        doc.close()
    return "\n\n".join(parts), pages, "ocr" if ocr_pages and ocr_pages * 2 >= max(1, pages) else "pdf-text"


def extract_docx(path: str) -> tuple[str, int, str]:
    from docx import Document

    doc = Document(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts), 0, "docx"


def extract_odt(path: str) -> tuple[str, int, str]:
    from odf.opendocument import load
    from odf.text import P

    doc = load(path)
    parts = []
    for elem in doc.getElementsByType(P):
        text = "".join(getattr(node, "data", None) or str(node) for node in elem.childNodes)
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts), 0, "odf"


def extract_plaintext(path: str) -> tuple[str, int, str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().strip(), 0, "text"


async def extract(path: str, mimetype: str | None, ocr: str, reader: Reader, scratch: str) -> tuple[str, int, str]:
    """A document's text, its page count and how it was read. Pictures are not documents here:
    `run_extract` reads them before it takes a tools slot."""
    if mimetype == "application/pdf":
        return await extract_pdf(path, ocr, reader, scratch)
    if mimetype == DOCX:
        return await asyncio.to_thread(extract_docx, path)
    if mimetype == ODT:
        return await asyncio.to_thread(extract_odt, path)
    if mimetype and mimetype.startswith("text/"):
        return await asyncio.to_thread(extract_plaintext, path)
    raise EmsError(UNSUPPORTED, f"no text extractor for {mimetype or 'an unknown type'}")


async def run_extract(ctx: Context, req: DocumentRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    mimetype = await mimetype_of(ctx, req.file, path)
    reader = Reader(ctx, req)
    await progress.emit("extract", message=kind_of(mimetype))
    try:
        if mimetype and mimetype.startswith("image/"):
            # A picture holds a model slot and no tools slot: reading it is the reader's work.
            text, pages, method = ("" if req.ocr == "never" else await reader.read(path, mimetype, req.file.angle, req.file.mirror)), 0, "ocr"
        else:
            scratch = ctx.scratch_dir("extract")
            try:
                async with ctx.admission.slot("tools", interactive=req.options.interactive):
                    text, pages, method = await extract(path, mimetype, req.ocr, reader, scratch)
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
    except EmsError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified below
        raise EmsError(TOOL_FAILED, str(exc) or type(exc).__name__, permanent=is_permanent(exc)) from exc
    result = {"text": text, "pages": pages, "method": method, "word_count": len(text.split())}
    return Outcome(result, producer=reader.model if method == "ocr" and reader.model else method)


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/document/extract")
    async def document_extract(request: Request, body: DocumentRequest):
        return await run_op(request, lambda progress: run_extract(ctx, body, progress))
