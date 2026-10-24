"""Pure functions: probe normalization, fpcalc output, face boxes."""

from ems.ops.faces import to_fractions
from ems.ops.fingerprint import parse_fpcalc
from ems.ops.media import kind_of, normalize
from ems.tools.ffprobe import Probe, fraction, ratio
from ems.worker import classify, worker_env


def test_kind_of():
    assert kind_of("image/heic") == "image"
    assert kind_of("video/quicktime") == "video"
    assert kind_of("audio/mpeg") == "audio"
    assert kind_of("application/pdf") == "document"
    assert kind_of("text/plain") == "document"
    assert kind_of("application/zip") == "other"
    assert kind_of(None) == "other"


def test_normalize_video(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"0" * 10)
    exif = {"MIMEType": "video/mp4", "Rotation": 90, "Make": "Apple", "Model": "iPhone 12", "CreateDate": "2025:09:27 15:26:32", "DateTimeOriginal": "2025:09:27 15:26:30", "OffsetTimeOriginal": "+02:00",
            "GPSLatitude": 59.33, "GPSLongitude": 18.07, "GPSAltitude": 12, "SerialNumber": "AB\x00"}
    probe = Probe({
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "61.400000", "bit_rate": "31500000"},
        "streams": [
            {"codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160, "sample_aspect_ratio": "1:1", "avg_frame_rate": "30000/1001", "field_order": "progressive",
             "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
            {"codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000"},
        ],
    }, None)
    out = normalize(str(f), exif, probe, None)
    assert out["mimetype"] == "video/mp4" and out["kind"] == "video" and out["size"] == 10
    assert out["container"].startswith("mov") and out["duration"] == 61.4 and out["bitrate"] == 31500000
    assert out["picture"] == {"width": 3840, "height": 2160, "codec": "hevc", "sar": 1, "fps": 29.97, "interlaced": False, "matrix_rotation": -90, "rotation": 90}
    assert out["sound"] == {"codec": "aac", "channels": 2, "sample_rate": 48000}
    assert out["captured_at"][0] == {"value": "2025:09:27 15:26:30", "source": "DateTimeOriginal", "zone": "+02:00"}
    assert out["captured_at"][1] == {"value": "2025:09:27 15:26:32", "source": "CreateDate"}
    assert out["gps"] == {"lat": 59.33, "lon": 18.07, "alt": 12.0}
    assert out["device"] == {"make": "Apple", "model": "iPhone 12", "serial": "AB"}
    assert out["decodable"] is True


def test_normalize_image_orientation_and_refusal(tmp_path):
    f = tmp_path / "i.jpg"
    f.write_bytes(b"0")
    out = normalize(str(f), {"MIMEType": "image/jpeg", "ImageWidth": 4000, "ImageHeight": 3000, "Orientation": 6}, None, None)
    assert out["picture"] == {"width": 4000, "height": 3000, "rotation": 90, "orientation": 6}
    out = normalize(str(f), {"MIMEType": "image/jpeg", "Orientation": 5}, None, None)
    assert out["picture"]["rotation"] == 90 and out["picture"]["mirrored"] is True
    refused = normalize(str(f), {"MIMEType": "video/quicktime"}, Probe(None, "moov atom not found"), None)
    assert refused["decodable"] is False and refused["reason"] == "moov atom not found"


def test_ffprobe_helpers():
    assert fraction("30000/1001") == 30000 / 1001
    assert fraction("0/0") is None and fraction("N/A") is None and fraction(None) is None
    assert ratio("4:3") == 4 / 3 and ratio("1:1") is None and ratio("0:1") is None


def test_fpcalc_values():
    assert parse_fpcalc("DURATION=10\nFINGERPRINT=1,2,4294967295\n") == [1, 2, 4294967295]
    assert parse_fpcalc("DURATION=10\n") == []


def test_face_fraction_conversion_filters_small_faces():
    raw = {"frame": {"width": 100, "height": 50}, "faces": [{"box": {"x": 10, "y": 5, "width": 10, "height": 10}, "confidence": 1, "embedding": []}]}
    assert to_fractions(raw, 0.0)["faces"][0]["box"] == {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.2}
    assert to_fractions(raw, 0.3)["faces"] == []
    assert to_fractions({"frame": {}, "faces": []}, 0)["faces"] == []
    # A malformed pose is dropped rather than failing the detection.
    raw["faces"][0]["pose"] = {"pitch": 1, "yaw": "x"}
    assert "pose" not in to_fractions(raw, 0.0)["faces"][0]


def test_worker_classification_and_env():
    assert classify("whisper waits for the card: 900MB free", "CardBusy").code.name == "model_unavailable"
    assert classify("transcribe: request expired", "TimeoutError").code.name == "deadline_exceeded"
    assert classify("caller hung up while waiting for the model", "RequestAbandoned").code.name == "deadline_exceeded"
    assert classify("audio not found: /x", "FileNotFoundError").code.name == "not_found"
    assert classify("could not load image: /x", "ValueError").code.name == "unsupported"
    assert classify("transcribe: missing path", "ValueError").code.name == "invalid_request"
    assert classify("weird", None).code.name == "tool_failed"
    assert classify("Required package not installed: torch", "ImportError").code.name == "internal"
    assert classify("unknown op: x", "ValueError").code.name == "invalid_request"
    # Running out of the card waits; it says nothing about the request.
    oom = classify("CUDA out of memory. Tried to allocate 2.00 GiB", "OutOfMemoryError")
    assert oom.code.name == "model_unavailable" and not oom.permanent and oom.retry_after
    assert classify("CUDA failed with error out of memory", "RuntimeError").code.name == "model_unavailable"
    assert classify("Failed to allocate memory for requested buffer of size 12", "RuntimeException").code.name == "model_unavailable"
    assert classify("cudaErrorInvalidDevice: invalid device ordinal", "RuntimeError").code.name == "model_unavailable"
    # A message that happens to say "not found" or "missing" is not the request's fault.
    lib = classify("Library libcublas.so.12 is not found or cannot be loaded", "RuntimeError")
    assert lib.code.name == "tool_failed" and not lib.permanent
    assert classify("cudnn missing", "RuntimeError").code.name == "tool_failed"
    env = worker_env({"EMS_WHISPER_LANGUAGE": "x", "CFG_WHISPER_MODEL": "y", "PATH": "/bin"})
    assert env["CFG_WHISPER_LANGUAGE"] == "x" and env["CFG_WHISPER_MODEL"] == "y" and env["EMS_WHISPER_LANGUAGE"] == "x"
