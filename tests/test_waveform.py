"""`audio.waveform`: what is drawn for each target, and one decode for all of them."""

import os

from ems.ops import audio, image
from ems.tools import magick


def target(width: int, height: int, fit: str = "contain") -> image.RenditionTarget:
    return image.RenditionTarget.model_validate({"output": {"path": f"/rw/{width}x{height}.avif"}, "box": {"width": width, "height": height}, "fit": fit})


def test_cover_is_drawn_square_and_contain_as_a_banner():
    assert audio.drawing_box(target(320, 320, "cover"), 3) == (320, 320)
    assert audio.drawing_box(target(512, 512), 3) == (512, 170)
    assert audio.drawing_box(target(512, 100), 3) == (512, 100)


def test_one_decode_draws_every_picture():
    graph = audio.filter_graph([(320, 320), (512, 170)], audio.WaveformStyle())
    assert graph.startswith("[0:a]aformat=channel_layouts=mono,aresample=8000,asplit=2[a0][a1];")
    assert "[a0]showwavespic=s=640x640:colors=#4a90f0:filter=average:scale=sqrt[fg0]" in graph
    assert "[a1]showwavespic=s=1024x340:colors=#4a90f0:filter=average:scale=sqrt[fg1]" in graph
    assert "color=s=640x640:color=#0a0b0e[bg0]" in graph
    assert "[bg1][fg1]overlay=format=rgb,drawbox=y=(ih-1)/2:w=iw:h=1:color=#2b70d6[out1]" in graph
    plain = audio.filter_graph([(320, 320)], audio.WaveformStyle(baseline=None, foreground="#ffffff"))
    assert "drawbox" not in plain and "colors=#ffffff" in plain


async def test_waveform_targets_are_fitted_from_their_own_drawing(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "tape.mp3").write_bytes(b"mp3")
    drawn = {}
    fitted = []

    async def fake_draw(ctx, path, scratch, boxes, style):
        drawn["boxes"], drawn["aspect"] = boxes, style.aspect
        outs = []
        for i in range(len(boxes)):
            out = os.path.join(scratch, f"wave-{i}.png")
            with open(out, "wb") as f:
                f.write(b"png")
            outs.append(out)
        return outs

    async def fake_render(source, angle, mirror, plans, *, timeout):
        fitted.append((os.path.basename(source), angle, plans[0].width, plans[0].height, plans[0].fit, plans[0].quality))
        with open(plans[0].temp, "wb") as f:
            f.write(b"avif")
        return magick.Rendered((plans[0].width * 2, plans[0].height * 2), [(plans[0].width, plans[0].height)])

    monkeypatch.setattr(audio, "draw", fake_draw)
    monkeypatch.setattr(magick, "render", fake_render)
    r = await client.post("/v0/audio/waveform", json={
        "file": {"path": str(ro / "tape.mp3")},
        "targets": [
            {"output": {"path": str(rw / "320x320.avif"), "quality": 55}, "box": {"width": 320, "height": 320}, "fit": "cover"},
            {"output": {"path": str(rw / "512x512.avif"), "quality": 58}, "box": {"width": 512, "height": 512}},
        ],
        "style": {"aspect": 4},
    })
    body = r.json()
    assert r.status_code == 200, body
    assert drawn == {"boxes": [(320, 320), (512, 128)], "aspect": 4}
    assert fitted == [("wave-0.png", 0, 320, 320, "contain", 55), ("wave-1.png", 0, 512, 128, "contain", 58)]
    assert body["result"]["targets"] == [
        {"path": str(rw / "320x320.avif"), "width": 320, "height": 320, "bytes": 4},
        {"path": str(rw / "512x512.avif"), "width": 512, "height": 128, "bytes": 4},
    ]

    r = await client.post("/v0/audio/waveform", json={"file": {"path": str(ro / "tape.mp3")}, "targets": [{"output": {"path": str(rw / "x.avif")}}]})
    assert r.status_code == 400 and "needs a box" in r.json()["error"]["message"]
