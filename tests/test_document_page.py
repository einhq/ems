"""`document.renditions`: the scale a page is rendered at, a text file's head, the first page of a
PDF and of a converted document, and what is refused. ImageMagick is faked; LibreOffice is real
where the machine has it."""

import os
import shutil

import pytest

from ems.ops import document_page
from ems.ops.image import BoxSize, RenditionTarget
from ems.schemas import OutputSpec
from ems.tools import magick


def target(width, height, fit="contain"):
    return RenditionTarget(output=OutputSpec(path="/rw/x.avif"), box=BoxSize(width=width, height=height), fit=fit)


def test_the_page_is_rendered_at_the_scale_the_largest_box_needs():
    a4 = (595.0, 842.0)
    # A contain box is filled by the page's longer fit: 1600 / 842.
    assert document_page.render_scale(*a4, [target(512, 512), target(1600, 1600)]) == pytest.approx(1600 / 842)
    # A cover box needs its short side covered: 320 / 595.
    assert document_page.render_scale(*a4, [target(320, 320, "cover")]) == pytest.approx(320 / 595)
    # A long strip never renders past MAX_SIDE.
    assert document_page.render_scale(100.0, 10000.0, [target(320, 320, "cover")]) == pytest.approx(document_page.MAX_SIDE / 10000)


def test_a_text_head_is_utf8_or_else_windows_1252(tmp_path):
    out = tmp_path / "out.txt"
    (tmp_path / "utf8.txt").write_bytes("Kära Anna, åter hemma.".encode("utf-8"))
    document_page.text_head(str(tmp_path / "utf8.txt"), str(out))
    assert out.read_text(encoding="utf-8") == "Kära Anna, åter hemma."
    (tmp_path / "latin.txt").write_bytes("Kära Anna".encode("cp1252"))
    document_page.text_head(str(tmp_path / "latin.txt"), str(out))
    assert out.read_text(encoding="utf-8") == "Kära Anna"
    # A character cut by the head's end is dropped, not read as Windows-1252.
    cut = ("a" * (document_page.TEXT_HEAD_BYTES - 1) + "å").encode("utf-8")
    (tmp_path / "long.txt").write_bytes(cut)
    document_page.text_head(str(tmp_path / "long.txt"), str(out))
    assert out.read_text(encoding="utf-8") == "a" * (document_page.TEXT_HEAD_BYTES - 1)


def _pdf(path, size=(200, 283)):
    pytest.importorskip("pypdfium2")
    image = pytest.importorskip("PIL.Image")
    pages = [image.new("L", size, 255), image.new("L", size, 200)]
    pages[0].save(str(path), "PDF", save_all=True, append_images=pages[1:], resolution=72)


@pytest.fixture
def fake_magick(monkeypatch):
    seen = {}

    async def render(source, angle, mirror, plans, *, timeout):
        from PIL import Image

        with Image.open(source) as picture:
            seen["size"] = picture.size
        seen["angle"] = angle
        for plan in plans:
            with open(plan.temp, "wb") as f:
                f.write(b"avif" * 10)
        return magick.Rendered(seen["size"], [seen["size"] for _ in plans])

    monkeypatch.setattr(magick, "render", render)
    return seen


async def test_a_pdfs_first_page_is_cut_into_every_target(client, dirs, fake_magick):
    ro, rw = dirs
    _pdf(ro / "letter.pdf")
    body = {
        "file": {"path": str(ro / "letter.pdf"), "mimetype": "application/pdf"},
        "targets": [
            {"output": {"path": str(rw / "320x320.avif")}, "box": {"width": 320, "height": 320}, "fit": "cover"},
            {"output": {"path": str(rw / "800x800.avif")}, "box": {"width": 800, "height": 800}},
        ],
    }
    r = await client.post("/v0/document/renditions", json=body)
    result = r.json()
    assert r.status_code == 200, result
    assert result["result"]["pages"] == 2
    assert result["result"]["page"] == {"width": 200, "height": 283}
    # 800 / 283 points: the contain box is the larger need.
    assert fake_magick["size"][1] == 800 and abs(fake_magick["size"][0] - 200 * 800 / 283) <= 1 and fake_magick["angle"] == 0
    assert sorted(os.listdir(rw)) == ["320x320.avif", "800x800.avif"]
    assert result["meta"]["producer"].startswith("pypdfium2/")

    r = await client.post("/v0/document/renditions", json={**body, "pdf": {"output": {"path": str(rw / "view.pdf")}}})
    assert r.status_code == 400 and "PDF already" in r.json()["error"]["message"]


async def test_what_is_no_document_or_no_pdf_is_refused(client, dirs, fake_magick):
    ro, rw = dirs
    (ro / "broken.pdf").write_bytes(b"%PDF-1.4 not really")
    (ro / "song.mp3").write_bytes(b"ID3")
    one = [{"output": {"path": str(rw / "320x320.avif")}, "box": {"width": 320, "height": 320}}]
    r = await client.post("/v0/document/renditions", json={"file": {"path": str(ro / "song.mp3"), "mimetype": "audio/mpeg"}, "targets": one})
    assert r.status_code == 415
    pytest.importorskip("pypdfium2")
    r = await client.post("/v0/document/renditions", json={"file": {"path": str(ro / "broken.pdf"), "mimetype": "application/pdf"}, "targets": one})
    assert r.status_code == 422 and r.json()["error"]["code"] == "undecodable"
    assert os.listdir(rw) == []


@pytest.mark.skipif(not shutil.which("soffice"), reason="LibreOffice is not installed")
async def test_a_text_file_is_converted_and_its_pdf_published(client, dirs, fake_magick):
    pytest.importorskip("pypdfium2")
    ro, rw = dirs
    (ro / "note.txt").write_text("Kära Anna,\n\nVi kommer på lördag.\n", encoding="utf-8")
    r = await client.post("/v0/document/renditions", json={
        "file": {"path": str(ro / "note.txt"), "mimetype": "text/plain"},
        "targets": [{"output": {"path": str(rw / "512x512.avif")}, "box": {"width": 512, "height": 512}}],
        "pdf": {"output": {"path": str(rw / "view.pdf")}},
    })
    result = r.json()
    assert r.status_code == 200, result
    assert result["result"]["pages"] == 1
    assert result["result"]["pdf"]["path"] == str(rw / "view.pdf") and result["result"]["pdf"]["bytes"] > 0
    assert (rw / "view.pdf").read_bytes().startswith(b"%PDF")
    assert max(fake_magick["size"]) == 512
    assert sorted(os.listdir(rw)) == ["512x512.avif", "view.pdf"]
    assert "soffice/" in result["meta"]["producer"]

    # The PDF alone, the page not rendered.
    os.remove(rw / "view.pdf")
    fake_magick.clear()
    r = await client.post("/v0/document/renditions", json={"file": {"path": str(ro / "note.txt"), "mimetype": "text/plain"}, "pdf": {"output": {"path": str(rw / "view.pdf")}}})
    result = r.json()
    assert r.status_code == 200, result
    assert result["result"]["targets"] == [] and result["result"]["pages"] == 1 and "frame" not in result["result"]
    assert (rw / "view.pdf").read_bytes().startswith(b"%PDF") and fake_magick == {}
    r = await client.post("/v0/document/renditions", json={"file": {"path": str(ro / "note.txt"), "mimetype": "text/plain"}})
    assert r.status_code == 400


@pytest.mark.skipif(not shutil.which("soffice"), reason="LibreOffice is not installed")
async def test_a_document_libreoffice_cannot_read_is_undecodable(client, dirs, fake_magick):
    ro, rw = dirs
    (ro / "broken.docx").write_bytes(b"PK\x03\x04 not a zip")
    r = await client.post("/v0/document/renditions", json={
        "file": {"path": str(ro / "broken.docx"), "mimetype": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        "targets": [{"output": {"path": str(rw / "512x512.avif")}, "box": {"width": 512, "height": 512}}],
        "pdf": {"output": {"path": str(rw / "view.pdf")}},
    })
    assert r.status_code == 422 and r.json()["error"]["code"] == "undecodable", r.json()
    assert os.listdir(rw) == []
