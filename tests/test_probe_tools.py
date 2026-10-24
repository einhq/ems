"""media.probe against the real exiftool/ffprobe when they happen to be installed."""

import shutil
import subprocess

import pytest

needs_tools = pytest.mark.skipif(not (shutil.which("exiftool") and shutil.which("convert")), reason="exiftool and ImageMagick not installed")


@needs_tools
async def test_probe_a_real_jpeg(client, roots):
    ro, _ = roots
    path = ro / "red.jpg"
    subprocess.run(["convert", "-size", "8x4", "xc:red", str(path)], check=True)
    r = await client.post("/v0/media/probe", json={"file": {"path": str(path)}, "hash": True})
    body = r.json()
    assert r.status_code == 200, body
    result = body["result"]
    assert result["mimetype"] == "image/jpeg" and result["kind"] == "image"
    assert result["picture"]["width"] == 8 and result["picture"]["height"] == 4
    assert result["decodable"] is True and len(result["sha256"]) == 64
    assert result["raw"]["exiftool"]["MIMEType"] == "image/jpeg"
    assert "SourceFile" not in result["raw"]["exiftool"]
    assert "ffprobe" not in result["raw"]  # an image needs no stream probe
    assert body["meta"]["producer"].startswith("exiftool/")

    # The stay-open process answers the second request too, and `raw` can be narrowed.
    r = await client.post("/v0/media/probe", json={"file": {"path": str(path)}, "raw": []})
    assert r.json()["result"]["raw"] == {}


@needs_tools
async def test_probe_streams_progress(client, roots):
    ro, _ = roots
    path = ro / "red.jpg"
    subprocess.run(["convert", "-size", "8x4", "xc:red", str(path)], check=True)
    async with client.stream("POST", "/v0/media/probe", json={"file": {"path": str(path)}}, headers={"Accept": "text/event-stream"}) as r:
        body = "".join([chunk async for chunk in r.aiter_text()])
    assert body.startswith("event: progress\ndata: {\"phase\":\"metadata\"}") and "event: result" in body


DATA = __import__("os").path.join(__import__("os").path.dirname(__file__), "data")
needs_exiftool = pytest.mark.skipif(not shutil.which("exiftool"), reason="exiftool not installed")


@needs_exiftool
async def test_probe_video_fixture_without_ffprobe_still_answers(client, roots):
    """Without ffprobe the video's shape and duration come from exiftool alone; with it, the streams too."""
    ro, _ = roots
    shutil.copy(f"{DATA}/video1.mp4", ro / "video1.mp4")
    r = await client.post("/v0/media/probe", json={"file": {"path": str(ro / "video1.mp4")}})
    body = r.json()
    assert r.status_code == 200, body
    result = body["result"]
    assert result["kind"] == "video" and result["duration"] == pytest.approx(5.568, abs=0.1)
    assert result["picture"]["width"] == 560 and result["picture"]["height"] == 320
    assert result["captured_at"][0]["source"] == "CreateDate"
    if shutil.which("ffprobe"):
        assert result["decodable"] is True and "ffprobe" in result["raw"] and result["sound"]["codec"]
    else:
        assert "ffprobe" not in result["raw"] and body["meta"]["warnings"]
