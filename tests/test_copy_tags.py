"""`image.copy_tags`: a picture given another file's metadata, its turn as its orientation, and the
edits made to it, with its pixels untouched."""

import json
import os
import shutil
import subprocess

import pytest

ICC = "/usr/share/color/icc/colord/AdobeRGB1998.icc"

pytestmark = pytest.mark.skipif(not (shutil.which("exiftool") and shutil.which("convert")), reason="exiftool and ImageMagick not installed")


def exiftool(*args):
    return subprocess.run(["exiftool", "-m", "-q", "-q", *args], check=True, capture_output=True).stdout


def tags(path):
    return json.loads(exiftool("-j", "-struct", "-G1", "-n", str(path)))[0]


def original(ro):
    """A camera JPEG: a date, a camera, a place, a caption, an edit of its own, a thumbnail, a colour
    profile and a quarter turn owed."""
    path = ro / "print.jpg"
    subprocess.run(["convert", "-size", "64x48", "gradient:red-blue", str(path)], check=True)
    subprocess.run(["convert", "-size", "16x12", "xc:red", str(ro / "thumb.jpg")], check=True)
    exiftool(
        "-overwrite_original", "-DateTimeOriginal=1974:06:01 12:00:00", "-Make=Canon", "-Model=AE-1", "-Orientation#=6", "-XMP-tiff:Orientation#=6",
        "-GPSLatitude=59.3", "-GPSLatitudeRef=N", "-GPSLongitude=18.1", "-GPSLongitudeRef=E", "-XMP-dc:Description=A sailboat",
        "-XMP-xmpMM:History={Action=created,When=2020:01:01 00:00:00,SoftwareAgent=Scanner 1.0}", "-XMP-xmpMM:InstanceID=xmp.iid:print",
        f"-ThumbnailImage<={ro / 'thumb.jpg'}", *([f"-ICC_Profile<={ICC}"] if os.path.exists(ICC) else []), str(path),
    )
    return path


def copy(path, ro, ext):
    """The picture's pixels, as a restoration op writes them: without any metadata."""
    out = ro / f"restored.{ext}"
    subprocess.run(["convert", str(path), "-strip", str(out)], check=True)
    exiftool("-overwrite_original", "-all=", str(out))
    return out


def pixels(path):
    return subprocess.run(["convert", str(path), "rgb:-"], check=True, capture_output=True).stdout


@pytest.mark.parametrize("ext", ["png", "tif"])
async def test_the_copy_carries_the_originals_tags_its_own_turn_and_its_edits(client, dirs, ext):
    ro, rw = dirs
    source = original(ro)
    file = copy(source, ro, ext)
    r = await client.post("/v0/image/copy_tags", json={
        "file": {"path": str(file), "angle": 90},
        "source": {"path": str(source)},
        "history": [{"action": "edited", "when": "2026-11-12T10:00:00Z", "software_agent": "unfade/1+opencv/4.10", "parameters": "image.unfade {clip: 0.5}"}],
        "digital_source_type": "algorithmicallyEnhanced",
        "output": {"path": str(rw / f"tagged.{ext}")},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    out = rw / f"tagged.{ext}"
    assert body["result"] == {"path": str(out), "bytes": out.stat().st_size}
    assert body["meta"]["producer"].startswith("exiftool/")
    got = tags(out)
    assert got["ExifIFD:DateTimeOriginal"] == "1974:06:01 12:00:00"
    assert got["IFD0:Make"] == "Canon" and got["IFD0:Model"] == "AE-1"
    assert got["Composite:GPSLatitude"] == pytest.approx(59.3) and got["Composite:GPSLongitude"] == pytest.approx(18.1)
    assert got["XMP-dc:Description"] == "A sailboat"
    if os.path.exists(ICC):
        assert "Adobe RGB" in got["ICC_Profile:ProfileDescription"]
    # The file's turn, 90° counter-clockwise, written as its only orientation.
    assert got["IFD0:Orientation"] == 8 and "XMP-tiff:Orientation" not in got
    assert "IFD1:ThumbnailImage" not in got and "XMP-xmpMM:InstanceID" not in got
    assert [event["Action"] for event in got["XMP-xmpMM:History"]] == ["created", "edited"]
    assert got["XMP-xmpMM:History"][1]["SoftwareAgent"] == "unfade/1+opencv/4.10"
    assert got["XMP-xmpMM:History"][1]["Parameters"] == "image.unfade {clip: 0.5}"
    assert got["XMP-iptcExt:DigitalSourceType"] == "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced"
    assert pixels(out) == pixels(file)


async def test_an_upright_copy_has_no_orientation_and_no_history_asked_adds_none(client, dirs):
    ro, rw = dirs
    source = original(ro)
    file = copy(source, ro, "png")
    r = await client.post("/v0/image/copy_tags", json={
        "file": {"path": str(file)}, "source": {"path": str(source)}, "output": {"path": str(rw / "tagged.png")},
    })
    assert r.status_code == 200, r.text
    got = tags(rw / "tagged.png")
    assert not any(key.endswith(":Orientation") for key in got)
    assert [event["Action"] for event in got["XMP-xmpMM:History"]] == ["created"]
    assert "XMP-iptcExt:DigitalSourceType" not in got


async def test_a_source_without_tags_gives_a_copy_without_them(client, dirs):
    ro, rw = dirs
    source = ro / "bare.png"
    subprocess.run(["convert", "-size", "8x8", "xc:blue", str(source)], check=True)
    exiftool("-overwrite_original", "-all=", str(source))
    r = await client.post("/v0/image/copy_tags", json={
        "file": {"path": str(source)}, "source": {"path": str(source)},
        "history": [{"action": "edited"}], "output": {"path": str(rw / "out.png")},
    })
    assert r.status_code == 200, r.text
    assert [event["Action"] for event in tags(rw / "out.png")["XMP-xmpMM:History"]] == ["edited"]


@pytest.mark.parametrize("change", [
    {"output": {"format": "png"}},
    {"output": {"path": "tagged.jpg"}},
    {"source": {"angle": 90}},
    {"digital_source_type": "http://cv.iptc.org/x"},
])
async def test_refusals(client, dirs, change):
    ro, rw = dirs
    source = original(ro)
    file = copy(source, ro, "png")
    body = {"file": {"path": str(file)}, "source": {"path": str(source)}, "output": {"path": str(rw / "tagged.png")}}
    for key, value in change.items():
        if key == "output" and "path" in value:
            value = {"path": str(rw / value["path"])}
        body[key] = {**body[key], **value} if isinstance(value, dict) else value
    r = await client.post("/v0/image/copy_tags", json=body)
    assert r.status_code in (400, 422), r.text
