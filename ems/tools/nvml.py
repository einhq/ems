"""The card's load over the last seconds, from the driver's own samples through NVML.

The driver records the card's compute, decoder and encoder load about every 200 ms and keeps the
last ~15 s of them. Their mean is the load over that whole stretch, where `nvidia-smi`'s
`utilization.gpu` is a single reading of the last sample period. NVML is the library `nvidia-smi`
is built on; this calls it in process through ctypes, with no fork and no package to install.
"""

from __future__ import annotations

import ctypes
import time

# nvmlSamplingType_t
GPU_UTILIZATION_SAMPLES = 1
ENC_UTILIZATION_SAMPLES = 3
DEC_UTILIZATION_SAMPLES = 4
# nvmlValueType_t: utilization samples are unsigned ints.
VALUE_TYPE_UNSIGNED_INT = 1

KINDS = (("util_pct", GPU_UTILIZATION_SAMPLES), ("dec_pct", DEC_UTILIZATION_SAMPLES), ("enc_pct", ENC_UTILIZATION_SAMPLES))

# How far back a load reaches: a caller polling health every 15 s sees all of the card's time.
WINDOW_SECONDS = 15.0


class _Sample(ctypes.Structure):
    # nvmlSample_t: a microsecond wall-clock timestamp and an 8-byte value union.
    _fields_ = [("ts", ctypes.c_ulonglong), ("value", ctypes.c_ulonglong)]


_device: tuple[ctypes.CDLL, ctypes.c_void_p] | None = None
_tried = False


def _open() -> tuple[ctypes.CDLL, ctypes.c_void_p] | None:
    """The library and the first card, opened once; None without a driver or a card."""
    global _device, _tried
    if _tried:
        return _device
    _tried = True
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        handle = ctypes.c_void_p()
        if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0:
            _device = (lib, handle)
    except (OSError, AttributeError):
        pass
    return _device


def _samples(lib: ctypes.CDLL, handle: ctypes.c_void_p, kind: int) -> list[tuple[int, int]] | None:
    """Every sample of `kind` the driver holds, as (timestamp µs, percent)."""
    value_type, count = ctypes.c_int(), ctypes.c_uint()
    if lib.nvmlDeviceGetSamples(handle, kind, ctypes.c_ulonglong(0), ctypes.byref(value_type), ctypes.byref(count), None) != 0:
        return None
    # Room for a few more than were counted: the driver may add one between the two calls.
    buffer = (_Sample * (count.value + 8))()
    count.value = len(buffer)
    if lib.nvmlDeviceGetSamples(handle, kind, ctypes.c_ulonglong(0), ctypes.byref(value_type), ctypes.byref(count), buffer) != 0:
        return None
    if value_type.value != VALUE_TYPE_UNSIGNED_INT:
        return None
    return [(sample.ts, sample.value & 0xFFFFFFFF) for sample in buffer[: count.value]]


def mean_since(samples: list[tuple[int, int]], since_us: int) -> float | None:
    """The mean percent of the samples taken at or after `since_us`, or None when there are none."""
    values = [value for ts, value in samples if ts >= since_us]
    return round(sum(values) / len(values), 1) if values else None


def load(window: float = WINDOW_SECONDS) -> dict | None:
    """`util_pct`, `dec_pct` and `enc_pct` averaged over the last `window` seconds, or None when
    NVML cannot say. A kind the card keeps no samples of is left out."""
    device = _open()
    if device is None:
        return None
    since_us = int((time.time() - window) * 1e6)
    out: dict = {}
    for key, kind in KINDS:
        samples = _samples(*device, kind)
        mean = mean_since(samples, since_us) if samples else None
        if mean is not None:
            out[key] = mean
    return out if "util_pct" in out else None
