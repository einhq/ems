"""`document.renditions` — a document's first page as picture renditions, and the document as PDF.

A PDF's first page is rendered by PDFium; any other document — Word, old Word, RTF, OpenDocument,
plain text — is converted to PDF by LibreOffice first, then rendered the same way. The page is
rendered once, at the scale the largest target needs, and every target is cut from that picture as
`image.renditions` cuts them. `pdf` asks for the converted PDF as well, which a viewer shows with a
browser's own reader; a PDF has no conversion to give.

Every output is published only once all of them have bytes: the pictures through
`image.render_targets`, the PDF through a private temporary file beside it.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
from importlib import metadata

from fastapi import APIRouter, Request
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import EmsError, TOOL_FAILED, UNDECODABLE, UNSUPPORTED, invalid
from ..schemas import FileRef, OutputSpec, Strict, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools.run import run, which
from ..tools.versions import producer
from .image import MAX_TARGETS, RenditionTarget, file_mimetype, render_targets

PDF = "application/pdf"

# What LibreOffice converts, by the extension it is handed under: it reads the type off the bytes,
# and the name is what settles a plain text file.
CONVERTED = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/msword": "doc",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    "text/plain": "txt",
}

# A first page needs no more of a text file than this; a long log converted whole would take minutes.
TEXT_HEAD_BYTES = 64 * 1024

# The longest side the page is rendered at, whatever the targets ask: a poster-sized page at the
# scale a 320 px cover crop needs of its short side would otherwise be tens of thousands of pixels.
MAX_SIDE = 4096


class PdfOutput(Strict):
    output: OutputSpec


class DocumentRenditionsRequest(WithOptions):
    file: FileRef
    # Empty when only the PDF is asked for.
    targets: list[RenditionTarget] = Field(default_factory=list, max_length=MAX_TARGETS)
    pdf: PdfOutput | None = None


def render_scale(page_width: float, page_height: float, targets: list[RenditionTarget]) -> float:
    """The scale (pixels per point) the largest target needs of a page: a `contain` box its longer
    fit, a `cover` or `fill` box its shorter one, bounded by `MAX_SIDE`. A target without a box
    takes the page at 2x, 144 dpi."""
    scale = 0.0
    for target in targets:
        box = target.box
        if box is None or (box.width is None and box.height is None):
            scale = max(scale, 2.0)
            continue
        ratios = [side / extent for side, extent in ((box.width, page_width), (box.height, page_height)) if side]
        scale = max(scale, max(ratios) if target.fit != "contain" else min(ratios))
    return min(scale, MAX_SIDE / max(page_width, page_height))


def text_head(path: str, out: str) -> None:
    """The start of a text file, as UTF-8: read as UTF-8, else as Windows-1252, which decodes any
    byte. A character cut at the end of the head is dropped."""
    with open(path, "rb") as handle:
        head = handle.read(TEXT_HEAD_BYTES)
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError as exc:
        text = head[:exc.start].decode("utf-8") if exc.start >= len(head) - 3 else head.decode("cp1252", errors="replace")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text)


async def convert(ctx: Context, path: str, extension: str, scratch: str) -> str:
    """The document as PDF, by LibreOffice, in `scratch`. Each call has a profile of its own, so
    conversions run side by side. A document LibreOffice cannot load is `undecodable`."""
    if which("soffice") is None:
        raise EmsError(UNSUPPORTED, "no document converter (LibreOffice) in this image")
    source = os.path.join(scratch, f"document.{extension}")
    if extension == "txt":
        await asyncio.to_thread(text_head, path, source)
    else:
        os.symlink(path, source)
    outdir = os.path.join(scratch, "out")
    args = [
        "soffice", "--headless", "--norestore", "--nolockcheck", "--nodefault",
        f"-env:UserInstallation=file://{os.path.join(scratch, 'profile')}",
    ]
    if extension == "txt":
        args.append("--infilter=Text (encoded):UTF8,LF,Liberation Mono,")
    args += ["--convert-to", "pdf", "--outdir", outdir, source]
    done = await run(args, timeout=ctx.settings.tool_stall_seconds, env={**os.environ, "HOME": scratch})
    converted = os.path.join(outdir, "document.pdf")
    if done.code != 0:
        raise EmsError(TOOL_FAILED, f"soffice exited {done.code}: {done.tail()}")
    if not os.path.isfile(converted) or os.path.getsize(converted) == 0:
        # LibreOffice exits 0 on a file it cannot load and says so on its output.
        said = (done.stderr_text.strip() or done.stdout.decode(errors="replace").strip())[-400:]
        raise EmsError(UNDECODABLE, f"LibreOffice could not read the document: {said or 'no PDF written'}")
    return converted


def render_first_page(pdf: str, targets: list[RenditionTarget], out: str) -> tuple[int, float, float]:
    """The first page of `pdf` as a PNG at the scale the targets need, on white, unless there are
    none: the page count and the page's size in points."""
    import pypdfium2

    try:
        doc = pypdfium2.PdfDocument(pdf)
    except pypdfium2.PdfiumError as exc:
        raise EmsError(UNDECODABLE, f"PDFium could not open the document: {exc}") from exc
    try:
        pages = len(doc)
        if pages == 0:
            raise EmsError(UNDECODABLE, "the document has no pages")
        page = doc[0]
        try:
            width, height = page.get_size()
            if width <= 0 or height <= 0:
                raise EmsError(UNDECODABLE, "the first page has no size")
            if targets:
                page.render(scale=render_scale(width, height, targets), fill_color=(255, 255, 255, 255)).to_pil().save(out, "PNG")
        finally:
            page.close()
    finally:
        doc.close()
    return pages, width, height


async def document_renditions(ctx: Context, req: DocumentRenditionsRequest, progress: Progress) -> Outcome:
    path = remote.input(req.file.path)
    mimetype = await file_mimetype(ctx, path, req.file)
    if mimetype != PDF and mimetype not in CONVERTED:
        raise EmsError(UNSUPPORTED, f"no page renderer for {mimetype or 'an unknown type'}")
    if req.pdf is not None and mimetype == PDF:
        raise invalid("the file is a PDF already: there is no PDF to make of it")
    if not req.targets and req.pdf is None:
        raise invalid("asks for neither a picture nor a PDF")
    if req.pdf is not None and os.path.splitext(req.pdf.output.path)[1].lower() != ".pdf":
        raise invalid(f"{req.pdf.output.path}: a PDF is named .pdf")
    pdf_final = remote.output(req.pdf.output.path) if req.pdf is not None else None
    pdf_temp = os.path.join(os.path.dirname(pdf_final), f".ems-{secrets.token_hex(8)}") if pdf_final else None
    async with ctx.admission.slot("tools", interactive=req.options.interactive):
        scratch = ctx.scratch_dir("document")
        try:
            source = path
            if mimetype != PDF:
                await progress.emit("convert")
                source = await convert(ctx, path, CONVERTED[mimetype], scratch)
            await progress.emit("render")
            page_png = os.path.join(scratch, "page.png")
            pages, width, height = await asyncio.to_thread(render_first_page, source, req.targets, page_png)
            if pdf_temp:
                await asyncio.to_thread(shutil.copyfile, source, pdf_temp)
            result = await render_targets(ctx, page_png, 0, False, req.targets) if req.targets else {"targets": []}
            result["pages"] = pages
            result["page"] = {"width": round(width, 2), "height": round(height, 2)}
            if pdf_temp and pdf_final:
                size = os.path.getsize(pdf_temp)
                os.replace(pdf_temp, pdf_final)
                result["pdf"] = {"path": req.pdf.output.path, "bytes": size}
        finally:
            if pdf_temp:
                try:
                    os.remove(pdf_temp)
                except FileNotFoundError:
                    pass
            shutil.rmtree(scratch, ignore_errors=True)
    pdfium = f"pypdfium2/{metadata.version('pypdfium2')}"
    tools = await producer("magick") if mimetype == PDF else await producer("soffice", "magick")
    return Outcome(result, producer=f"{pdfium}+{tools}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/document/renditions")
    async def document_renditions_route(request: Request, body: DocumentRenditionsRequest):
        return await run_op(request, lambda progress: document_renditions(ctx, body, progress))
