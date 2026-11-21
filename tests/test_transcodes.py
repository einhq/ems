"""`transcodes`: a granted stream read as it comes and written as it is encoded into another the
caller grants, the encode started anew on the source after it drops, its output closed when it is
deleted, a transcode ended when its output cannot be written, and the ffmpeg it runs. The encode
is stood in for by a copy (`cat`)."""

import asyncio
import os
import tempfile

import pytest

from ems import transcodes
from ems.errors import EmsError, NOT_FOUND
from ems.transcodes import TranscodeSpec, encoder_args

SPEC = {"source": {"path": "/0f/0-main.ts"}, "output": {"path": "/0f/1-main.h264.ts"}}


class Names:
    """The caller's names: the source a pipe whose writing end the test holds, opened anew each
    time it is opened; the output a file of its own."""

    def __init__(self):
        self.folder = tempfile.mkdtemp(prefix="ems-transcode-")
        self.sources: list[int] = []
        self.opened: list[str] = []
        self.refuse = False

    async def source(self, endpoint, name):
        self.opened.append(name)
        if self.refuse:
            raise EmsError(NOT_FOUND, f"{name}: no such name")
        read, write = os.pipe()
        self.sources.append(write)
        return read

    async def output(self, endpoint, name):
        return os.open(os.path.join(self.folder, "out"), os.O_WRONLY | os.O_CREAT)

    def written(self) -> bytes:
        return open(os.path.join(self.folder, "out"), "rb").read()


@pytest.fixture
def names(client, monkeypatch):
    monkeypatch.setattr(transcodes, "RETRY_SECONDS", (0.05, 0.1))
    found = Names()
    ctx = client.app.state.ctx

    async def encoders():
        return {"h264_nvenc", "libopenh264"}

    ctx.transcodes.open_source = found.source
    ctx.transcodes.open_output = found.output
    ctx.transcodes.encoders = encoders
    # bash, since dash takes a single digit after `<&`.
    ctx.transcodes.command = lambda spec, encoder, fd: ["bash", "-c", f"cat <&{fd}"]
    return found


async def _until(done, what: str) -> None:
    for _ in range(200):
        if done():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never: {what}")


async def test_the_source_is_written_into_the_output_as_it_comes_and_again_after_it_drops(client, names):
    r = await client.put("/v0/transcodes/porch", json=SPEC)
    assert r.status_code == 200, r.text
    status = r.json()["result"]
    assert status["made"] is True and status["source"] == "/0f/0-main.ts" and status["encoder"] == "h264_nvenc"
    again = (await client.put("/v0/transcodes/porch", json=SPEC)).json()["result"]
    assert again["made"] is False
    await _until(lambda: names.sources, "the source is opened")
    os.write(names.sources[0], b"first ")
    await _until(lambda: names.written() == b"first ", "the first bytes are written")
    assert (await client.get("/v0/transcodes/porch")).json()["result"]["state"] == "running"
    # The source drops, and is opened again: what comes then follows on in the same output.
    os.close(names.sources[0])
    await _until(lambda: len(names.sources) == 2, "the source is opened again")
    os.write(names.sources[1], b"then")
    await _until(lambda: names.written() == b"first then", "the output follows on")
    assert [each["id"] for each in (await client.get("/v0/transcodes")).json()["result"]] == ["porch"]
    transcode = client.app.state.ctx.transcodes.get("porch")
    output = transcode.output
    assert (await client.delete("/v0/transcodes/porch")).status_code == 200
    with pytest.raises(OSError):
        os.fstat(output)
    assert (await client.get("/v0/transcodes/porch")).status_code == 404
    os.close(names.sources[1])


async def test_a_source_not_granted_is_tried_again_and_said_lost(client, names):
    names.refuse = True
    await client.put("/v0/transcodes/porch", json=SPEC)
    await _until(lambda: len(names.opened) >= 2, "the source is tried again")
    status = (await client.get("/v0/transcodes/porch")).json()["result"]
    assert status["state"] == "lost" and "no such name" in status["error"]
    await client.delete("/v0/transcodes/porch")


async def test_a_transcode_whose_output_cannot_be_written_ends(client, names):
    await client.put("/v0/transcodes/porch", json=SPEC)
    transcode = client.app.state.ctx.transcodes.get("porch")
    await _until(lambda: names.sources, "the source is opened")
    os.close(transcode.output)
    os.write(names.sources[0], b"lost")
    await _until(lambda: transcode.ending is not None, "the transcode ends")
    status = (await client.get("/v0/transcodes/porch")).json()["result"]
    assert status["state"] == "ended" and status["reason"].startswith("the output could not be written")
    os.close(names.sources[0])


async def test_a_transcode_is_put_with_its_endpoint_and_past_the_limit_is_busy(client, names):
    r = await client.put("/v0/transcodes/porch", json=SPEC, headers={"ems-files": ""})
    assert r.status_code == 400 and "EMS-Files" in r.json()["error"]["message"]
    assert (await client.put("/v0/transcodes/porch", json={"source": SPEC["source"]})).status_code == 400
    client.app.state.ctx.transcodes.limit = 1
    await client.put("/v0/transcodes/porch", json=SPEC)
    r = await client.put("/v0/transcodes/hall", json=SPEC)
    assert r.status_code == 503 and r.json()["error"]["code"] == "busy"
    await client.delete("/v0/transcodes/porch")


def test_the_encode_reads_the_source_as_a_pipe_and_writes_h264_with_a_keyframe_a_second():
    args = encoder_args(TranscodeSpec(**SPEC, video={"height": 720}), "h264_nvenc", 7)
    assert args[args.index("-i") + 1] == "pipe:7" and args[-2:] == ["mpegts", "pipe:1"]
    assert args[args.index("-i") - 2:args.index("-i")] == ["-protocol_whitelist", "pipe"]
    assert args[args.index("-c:v") + 1] == "h264_nvenc" and "expr:gte(t,n_forced*1)" in args
    assert "scale=-2:'min(ih,720)'" in args[args.index("-vf") + 1]
    assert args[args.index("-map", args.index("-map") + 1) + 1] == "0:a:0?"


def test_an_mp4_output_is_fragmented_at_each_keyframe():
    args = encoder_args(TranscodeSpec(**SPEC, format="mp4"), "libopenh264", 7)
    assert args[-2:] == ["mp4", "pipe:1"] and "frag_keyframe+empty_moov+default_base_moof" in args
    assert "dump_extra=freq=keyframe" not in args


async def test_a_source_that_fails_leaves_the_encoder_and_one_that_cannot_start_gives_way(client, names, monkeypatch):
    said = {"error": "Invalid data found when processing input"}
    # The stand-in writes nothing and says what ffmpeg would have.
    client.app.state.ctx.transcodes.command = lambda spec, encoder, fd: ["bash", "-c", f"echo '{said['error']}' >&2; exit 1"]
    await client.put("/v0/transcodes/porch", json=SPEC)
    transcode = client.app.state.ctx.transcodes.get("porch")
    await _until(lambda: len(names.opened) >= 2, "the source is tried again")
    assert transcode.encoder == "h264_nvenc"
    said["error"] = "[h264_nvenc @ 0x5] OpenEncodeSessionEx failed: out of memory (10)"
    await _until(lambda: transcode.encoder == "libopenh264", "the CPU's encoder takes over")
    await client.delete("/v0/transcodes/porch")

