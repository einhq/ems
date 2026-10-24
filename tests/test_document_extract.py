"""`document.extract`: what is refused before anything runs, and how pictures and scanned pages are read."""

import pytest

from ems.errors import EmsError, UNSUPPORTED
from ems.ops.document import extract


async def test_unknown_type_is_refused_permanently(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00")
    with pytest.raises(EmsError) as raised:
        await extract(str(f), "application/octet-stream", "auto", None, str(tmp_path))
    assert raised.value.code is UNSUPPORTED


async def test_a_picture_is_read_by_the_text_reader_in_its_display_frame(client, fake_worker, roots):
    ro, _ = roots
    (ro / "sign.jpg").write_bytes(b"jpeg")
    fake_worker.on("read_text", lambda req: {"text": "Hamngatan 4\nÖppet 10–18", "lines": [], "width": 10, "height": 10, "model": "PP-OCRv6_medium"})
    r = await client.post("/v0/document/extract", json={"file": {"path": str(ro / "sign.jpg"), "mimetype": "image/jpeg", "angle": 90}, "languages": ["swe"]})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"] == {"text": "Hamngatan 4\nÖppet 10–18", "pages": 0, "method": "ocr", "word_count": 4}
    assert body["meta"]["producer"] == "PP-OCRv6_medium"
    sent = fake_worker.requests[-1]
    assert sent["op"] == "read_text" and sent["paths"] == [str(ro / "sign.jpg")] and sent["angle"] == 90


async def test_ocr_never_reads_no_picture(client, fake_worker, roots):
    ro, _ = roots
    (ro / "sign.jpg").write_bytes(b"jpeg")
    body = (await client.post("/v0/document/extract", json={"file": {"path": str(ro / "sign.jpg"), "mimetype": "image/jpeg"}, "ocr": "never"})).json()
    assert body["result"]["text"] == "" and body["meta"]["producer"] == "ocr"
    assert not [q for q in fake_worker.requests if q["op"] == "read_text"]


def _scanned_pdf(path, pages: int):
    """A PDF whose pages are pictures with no text layer, as a scanner writes one."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=200, height=200)
        page.draw_rect(fitz.Rect(20, 20, 180, 60), color=(0, 0, 0), fill=(0.2, 0.2, 0.2))
    doc.save(str(path))


async def test_a_scanned_pdf_is_read_page_by_page(client, fake_worker, roots):
    ro, _ = roots
    _scanned_pdf(ro / "scan.pdf", 2)
    pages = iter(["Protokoll", "Sida två"])
    fake_worker.on("read_text", lambda req: {"text": next(pages), "lines": [], "width": 1, "height": 1, "model": "PP-OCRv6_medium"})
    body = (await client.post("/v0/document/extract", json={"file": {"path": str(ro / "scan.pdf"), "mimetype": "application/pdf"}})).json()
    assert body["ok"], body
    assert body["result"] == {"text": "Protokoll\n\nSida två", "pages": 2, "method": "ocr", "word_count": 3}
    assert body["meta"]["producer"] == "PP-OCRv6_medium"
    assert len([q for q in fake_worker.requests if q["op"] == "read_text"]) == 2


async def test_a_busy_reader_fails_the_document_rather_than_store_it_empty(client, fake_worker, roots):
    ro, _ = roots
    _scanned_pdf(ro / "scan.pdf", 1)

    def busy(req):
        raise RuntimeError("CUDA failed with error out of memory")

    fake_worker.on("read_text", busy)
    body = (await client.post("/v0/document/extract", json={"file": {"path": str(ro / "scan.pdf"), "mimetype": "application/pdf"}})).json()
    assert body["ok"] is False and body["error"]["permanent"] is False
