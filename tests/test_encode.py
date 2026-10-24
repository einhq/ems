"""The transcode plan: filter chains, the encoder arguments and the CPU decode budget."""

import pytest

from ems.tools import encode
from ems.ops import transcode

BUDGET = encode.Budget(4000, 170, 64)


def test_filter_chains_fit_never_upscale_and_rotate_then_flop():
    shape = encode.Shape(1920, 1080, angle=270, mirror=True, deinterlace=True)
    assert encode.cpu_filters(shape) == (
        "yadif,scale=w=min(iw\\,1920):h=min(ih\\,1080):force_original_aspect_ratio=decrease:force_divisible_by=2,transpose=1,hflip"
    )
    # No CUDA chain for a mirror: there is no hflip_cuda, and an approximation would put the two chains out of step.
    assert encode.cuda_filters(shape) is None
    plain = encode.Shape(480, 480, angle=90)
    assert encode.cuda_filters(plain) == (
        "scale_cuda=w=min(iw\\,480):h=min(ih\\,480):force_original_aspect_ratio=decrease:force_divisible_by=2:format=yuv420p,transpose_cuda=dir=2"
    )
    assert encode.cuda_filters(encode.Shape(angle=180)) == "scale_cuda=format=yuv420p,transpose_cuda=dir=4"
    assert encode.cpu_filters(encode.Shape(angle=180)) == "transpose=1,transpose=1"
    assert encode.cpu_filters(encode.Shape()) == "null"
    assert encode.upload_filters(encode.Shape(width=640)) == "scale=w=min(iw\\,640):h=-2,format=yuv420p,hwupload_cuda"


def test_encoder_arguments():
    assert encode.cpu_encode_args(32, 8) == ["-c:v", "libsvtav1", "-preset", "8", "-crf", "32", "-pix_fmt", "yuv420p"]
    assert encode.gpu_encode_args("av1_nvenc", 32, cq_offset=6, preset="p5") == ["-c:v", "av1_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "38"]


def test_a_source_within_budget_runs_whole():
    plan = encode.plan_cpu_decode(60, 25, 1, BUDGET, clipped_whole_seconds=True)
    assert plan == encode.Plan(frames=1500, projected_mb=255)


def test_a_long_source_is_windowed_and_the_count_rederived():
    # An 84-minute DV tape: 126,000 frames project to 21,420 MB.
    plan = encode.plan_cpu_decode(5040, 25, 1, BUDGET, clipped_whole_seconds=True)
    assert plan == encode.Plan(frames=126000, projected_mb=21420, windows=6, window_seconds=840)
    # 30000/1001 fps, rounded up on the guarded side.
    plan = encode.plan_cpu_decode(1000, 30000, 1001, BUDGET, clipped_whole_seconds=True)
    assert isinstance(plan, encode.Plan) and plan.frames == 29971 and plan.windows == 2 and plan.window_seconds == 500


def test_refusals():
    assert isinstance(encode.plan_cpu_decode(60, 25, 1, encode.Budget(0, 170, 64), clipped_whole_seconds=True), encode.Refusal)
    assert "invalid" in encode.plan_cpu_decode(60, 25, 1, encode.Budget(4000, 0, 64), clipped_whole_seconds=True).reason
    too_many = encode.plan_cpu_decode(5040 * 20, 25, 1, encode.Budget(4000, 170, 4), clipped_whole_seconds=True)
    assert isinstance(too_many, encode.Refusal) and "windows" in too_many.reason
    fractional = encode.plan_cpu_decode(5040, 25, 1, BUDGET, clipped_whole_seconds=False)
    assert isinstance(fractional, encode.Refusal) and "whole seconds" in fractional.reason
    # A second of 1000 fps costs 170 MB; at a 100 MB budget no whole-second window fits.
    dense = encode.plan_cpu_decode(10, 1000, 1, encode.Budget(100, 170, 64), clipped_whole_seconds=True)
    assert isinstance(dense, encode.Refusal) and "shortest whole-second window" in dense.reason


def test_seconds_to_decode_is_the_clip_or_the_remaining_source():
    assert encode.seconds_to_decode(100.2, start=0, clip_seconds=None) == 101
    assert encode.seconds_to_decode(100.2, start=40, clip_seconds=None) == 61
    assert encode.seconds_to_decode(100.2, start=1, clip_seconds=20) == 20
    assert encode.seconds_to_decode(None, start=0, clip_seconds=None) is None


def test_audio_plan_is_a_ceiling_and_remuxes_aac_in_mp4():
    spec = transcode.AudioTranscodeSpec(bitrate="128k", sample_rate=44100)
    args, facts = transcode.audio_plan({"codec_name": "mp3", "bit_rate": "64000", "sample_rate": "22050", "channels": 1}, "mp3", spec)
    assert args[:11] == ["-vn", "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "64000", "-ar", "22050", "-ac", "1"]
    assert args[-2:] == ["-f", "ipod"]
    assert facts == {"bitrate": 64000, "sample_rate": 22050, "channels": 1, "remuxed": False}
    args, facts = transcode.audio_plan({"codec_name": "flac", "bit_rate": "900000", "sample_rate": "96000", "channels": 2}, "flac", spec)
    assert "-b:a" in args and args[args.index("-b:a") + 1] == "128000" and args[args.index("-ar") + 1] == "44100" and "-ac" not in args
    args, facts = transcode.audio_plan({"codec_name": "aac", "bit_rate": "128000", "sample_rate": "44100", "channels": 2}, "mov,mp4,m4a,3gp,3g2,mj2", spec)
    assert args[:3] == ["-vn", "-c:a", "copy"] and facts["remuxed"] is True
    args, facts = transcode.audio_plan({"codec_name": "aac", "bit_rate": "128000", "sample_rate": "44100", "channels": 2}, "aac", spec)
    assert facts["remuxed"] is False
    with pytest.raises(Exception, match="bitrate"):
        transcode.bits_per_second("loud")
