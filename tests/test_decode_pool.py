"""Pictures are decoded on the decode pool, outside their family's lock: the next request's picture
is read while the model works on this one."""

import os
import sys
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import inference_ops  # noqa: E402
import model_registry as registry  # noqa: E402


def test_decode_all_waits_for_the_pins_and_answers_in_order(monkeypatch):
    monkeypatch.setattr(registry, "_decode_executor", None)
    ready = threading.Event()
    monkeypatch.setattr(registry, "pinned_ready", ready)

    out = []
    decode = threading.Thread(target=lambda: out.extend(registry.decode_all(lambda n: n * n, [1, 2, 3, 4, 5])))
    decode.start()
    decode.join(0.2)
    assert decode.is_alive() and out == []              # its first imports wait for the preload
    ready.set()
    decode.join(2)
    assert out == [1, 4, 9, 16, 25]


def test_a_picture_is_decoded_while_another_request_holds_the_family(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_family_locks", {})
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"jpeg")
    decoded = threading.Event()

    def open_display(path, angle, mirror, min_side=0):
        decoded.set()
        return f"pixels of {os.path.basename(path)}"

    monkeypatch.setitem(sys.modules, "image_io", types.SimpleNamespace(open_display=open_display))
    monkeypatch.setattr(registry, "copy_detect_device", lambda: "cpu")
    monkeypatch.setattr(registry, "sscd", lambda device: types.SimpleNamespace(describe=lambda images: [[len(images[0])]]))

    out = {}
    with registry.family_lock(inference_ops.SSCD):           # the model is busy with another picture
        request = threading.Thread(target=lambda: out.update(inference_ops.copy_descriptors([str(picture)])))
        request.start()
        assert decoded.wait(2)
        assert out == {}
    request.join(2)
    assert out["descriptors"] == [[len("pixels of a.jpg")]]
