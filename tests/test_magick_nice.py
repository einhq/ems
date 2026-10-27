import asyncio

from ems.tools import magick
from ems.tools.run import BACKGROUND_NICE


def test_render_runs_magick_at_background_nice(monkeypatch):
    seen = {}

    async def fake_run(args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(magick, "run", fake_run)
    monkeypatch.setattr(magick, "require", lambda name: name)
    try:
        asyncio.run(magick.render("in.jpg", 0, False, [], timeout=60))
    except RuntimeError:
        pass

    assert seen["niceness"] == BACKGROUND_NICE
