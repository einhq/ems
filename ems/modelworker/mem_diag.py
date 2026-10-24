"""Host-memory diagnostics for the resident model server.

Exists to answer one question: when the server's RSS ratchets up over hours, do the bytes
belong to glibc's allocator or not?

`malloc_trim(0)` (model_registry.cleanup_torch) can only return memory glibc owns. If
`arena + hblkhd` climbs with RSS, something is malloc'd and never freed and the caller is
findable. If glibc's total stays flat while RSS climbs, the bytes belong to an allocator that
mmaps for itself and never gives back — onnxruntime's CPU BFCArena, CUDA pinned host memory,
CTranslate2 — and no amount of trimming will help.

Everything here is best-effort: diagnostics must never take down an inference request, so every
probe returns None rather than raising.
"""

import os
import sys
import threading
import time

_libc = None

# mallinfo2 (glibc 2.33+) is size_t-wide. The older mallinfo returns ints, which wrap silently
# at 4GB — useless for a process we expect to reach 9GB, so a wrapped reading is reported as
# such rather than quietly believed.
_MALLINFO_FIELDS = (
    "arena",     # non-mmapped space allocated by malloc (sbrk heaps, all arenas)
    "ordblks",   # free chunks
    "smblks",    # free fastbin blocks
    "hblks",     # mmapped regions owned by malloc
    "hblkhd",    # space in those mmapped regions
    "usmblks",   # high-water (unused since glibc 2.x)
    "fsmblks",   # space in freed fastbin blocks
    "uordblks",  # total allocated space -- the "in use" number
    "fordblks",  # total free space held by the allocator
    "keepcost",  # top-most releasable space
)


def _get_libc():
    global _libc
    if _libc is None:
        import ctypes
        try:
            _libc = ctypes.CDLL("libc.so.6")
        except OSError:
            _libc = False
    return _libc or None


def mallinfo() -> dict | None:
    """glibc allocator totals across all arenas, in bytes. None off glibc."""
    libc = _get_libc()
    if libc is None:
        return None

    import ctypes

    wrapped = False
    fn = getattr(libc, "mallinfo2", None)
    if fn is not None:
        ctype = ctypes.c_size_t
    else:
        fn = getattr(libc, "mallinfo", None)
        if fn is None:
            return None
        # 32-bit counters: anything at/above 4GB has already wrapped.
        ctype = ctypes.c_int
        wrapped = True

    class _Mallinfo(ctypes.Structure):
        _fields_ = [(name, ctype) for name in _MALLINFO_FIELDS]

    fn.restype = _Mallinfo
    fn.argtypes = []
    try:
        info = fn()
    except Exception:  # noqa: BLE001 - diagnostics never raise
        return None

    out = {name: int(getattr(info, name)) for name in _MALLINFO_FIELDS}
    if wrapped:
        out["counters_may_have_wrapped"] = True
    return out


def _smaps_rollup() -> dict | None:
    """RSS/Anonymous/Swap for the whole process, in bytes."""
    try:
        with open("/proc/self/smaps_rollup", "r") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None

    wanted = {"Rss": "rss", "Anonymous": "anon", "Swap": "swap", "Pss_File": "file"}
    out: dict[str, int] = {}
    for line in lines:
        key, _, rest = line.partition(":")
        name = wanted.get(key)
        if name is not None:
            try:
                out[name] = int(rest.strip().split()[0]) * 1024
            except (ValueError, IndexError):
                pass
    return out or None


def _anon_mappings(top: int = 5) -> dict | None:
    """Count and size the anonymous mappings.

    An allocator that extends by mmap (ORT's BFCArena, CUDA) shows up here as a rising *count*
    of large anonymous regions, because an mmap region cannot grow in place. glibc arena growth
    instead shows as a handful of regions getting bigger. That difference is the tell.
    """
    try:
        with open("/proc/self/maps", "r") as handle:
            raw = handle.read().splitlines()
    except OSError:
        return None

    sizes: list[int] = []
    for line in raw:
        parts = line.split()
        # 5 fields = no pathname = anonymous. 6 fields with [heap]/[stack] are named regions
        # we deliberately leave out; they are not what an arena extends into.
        if len(parts) != 5:
            continue
        span = parts[0]
        start, _, end = span.partition("-")
        try:
            sizes.append(int(end, 16) - int(start, 16))
        except ValueError:
            continue

    if not sizes:
        return None
    sizes.sort(reverse=True)
    return {"count": len(sizes), "total": sum(sizes), "largest": sizes[:top]}


def _torch_cuda() -> dict | None:
    """torch's VRAM accounting, only if torch is already imported."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
        }
    except Exception:  # noqa: BLE001
        return None


def snapshot() -> dict:
    """One reading of every probe. Missing probes are simply absent."""
    out: dict[str, object] = {"at": round(time.time(), 1), "threads": threading.active_count()}
    for key, value in (
        ("proc", _smaps_rollup()),
        ("glibc", mallinfo()),
        ("anon_maps", _anon_mappings()),
        ("torch_cuda", _torch_cuda()),
    ):
        if value is not None:
            out[key] = value
    return out


def _mb(value: float | None) -> str:
    return "?" if value is None else f"{value / 1048576:.0f}"


def format_line(snap: dict, loaded: object = None) -> str:
    """Compact one-line rendering for the periodic log.

    Deliberately flat and greppable: this is read back as a time series out of
    `podman logs`, hours after the fact, by `grep memstats`.
    """
    proc = snap.get("proc") or {}
    glibc = snap.get("glibc") or {}
    maps = snap.get("anon_maps") or {}
    cuda = snap.get("torch_cuda") or {}

    # glibc's own total, the number that decides whether malloc_trim could ever help.
    heap = None
    if glibc:
        heap = glibc.get("arena", 0) + glibc.get("hblkhd", 0)

    parts = [
        f"rss={_mb(proc.get('rss'))}MB",
        f"anon={_mb(proc.get('anon'))}MB",
        f"glibc={_mb(heap)}MB",
        f"inuse={_mb(glibc.get('uordblks'))}MB",
        f"free={_mb(glibc.get('fordblks'))}MB",
        f"maps={maps.get('count', '?')}",
        f"mapsMB={_mb(maps.get('total'))}",
        f"threads={snap.get('threads', '?')}",
    ]
    if cuda:
        parts.append(f"vram_alloc={_mb(cuda.get('allocated'))}MB")
        parts.append(f"vram_resv={_mb(cuda.get('reserved'))}MB")
    if loaded:
        parts.append(f"loaded={','.join(sorted(loaded)) if not isinstance(loaded, str) else loaded}")
    return " ".join(parts)


def log_loop(stop: threading.Event, log, interval: float, loaded_fn=None) -> None:
    """Log a `memstats` line every `interval` seconds until `stop`.

    Logs once immediately so a restart always leaves a t=0 baseline in the series — without it
    the first data point is one interval in, and the load ramp is exactly the part that has to
    be told apart from the ratchet.
    """
    while True:
        try:
            loaded = None
            if loaded_fn is not None:
                try:
                    loaded = loaded_fn()
                except Exception:  # noqa: BLE001
                    pass
            log(f"memstats {format_line(snapshot(), loaded)}")
        except Exception as exc:  # noqa: BLE001 - a broken probe must not kill the thread
            log(f"memstats failed: {exc}")
        if stop.wait(interval):
            return


def interval_from_env() -> float:
    """`CFG_MEMLOG_INTERVAL` seconds; 0 or unset-to-0 disables the periodic log."""
    raw = os.environ.get("CFG_MEMLOG_INTERVAL", "300")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 300.0


# --- The numbers a self-imposed ceiling is made of --------------------------------------
#
# Readings only: what RSS is now, and what cap the process will be killed against. The policy
# built on them — when to cross, what to log, and the drain and re-exec — is the server's, in
# model_server.py.

def rss_bytes() -> int | None:
    """This process's resident set size, from the cheapest source there is.

    `/proc/self/statm` is one read of a counter the kernel already maintains, where
    `_smaps_rollup` walks every mapping. The watchdog samples this on a short interval and
    only wants the one number; the full `snapshot()` is taken once, when the ceiling is
    actually crossed and the breakdown becomes the evidence.
    """
    try:
        with open("/proc/self/statm", "r") as handle:
            fields = handle.read().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _mem_total() -> int | None:
    """Physical RAM in bytes, from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


_CGROUP_LIMIT_FILES = (
    "/sys/fs/cgroup/memory.max",                    # v2, what podman --memory writes
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
)


def memory_limit() -> tuple[int | None, str]:
    """The cap this process would be OOM-killed against, and where it was read.

    The cgroup cap is the number that matters — the container's memory limit is what the
    kernel enforces, not the host's RAM — but the server also runs uncontained on a dev box, where
    physical memory is the honest stand-in. cgroup v1 spells "unlimited" as a sentinel far
    larger than any real machine, so a limit above physical memory is read as no limit.
    """
    total = _mem_total()
    for path in _CGROUP_LIMIT_FILES:
        try:
            with open(path, "r") as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        if raw == "max":
            break
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or (total is not None and value > total):
            break
        return value, f"cgroup {path}"
    if total is not None:
        return total, "/proc/meminfo MemTotal (no cgroup limit)"
    return None, "unknown"


def _parse_size(raw: str) -> int | None:
    """`14g`, `14000m`, `1024k` or plain bytes. None when it is not a size at all."""
    text = raw.strip().lower().rstrip("b")
    if not text:
        return None
    scale = {"k": 1024, "m": 1048576, "g": 1073741824}.get(text[-1:])
    if scale is not None:
        text = text[:-1]
    try:
        return int(float(text) * (scale or 1))
    except ValueError:
        return None


# Fraction of the memory limit the server allows itself when no absolute ceiling is set.
#
# Sized from what else may live in the same cgroup rather than from the server alone. High
# enough that a legitimate model peak does not trip it, low enough that the recycle
# happens while the rest of the container still has room — the point is to act before the
# cgroup starts reclaiming, not to survive one more request.
DEFAULT_MAX_RSS_FRACTION = 0.6


def max_rss_from_env() -> tuple[int | None, str]:
    """The server's RSS ceiling in bytes and a phrase explaining where it came from.

    `CFG_MODEL_MAX_RSS` is absolute and wins when set (`14g`, `14000m`, or plain bytes);
    `0` disables the ceiling entirely. Otherwise it is `CFG_MODEL_MAX_RSS_FRACTION` of
    the `memory_limit()` above. None means no ceiling — either switched off, or nothing to
    take a fraction of.
    """
    raw = os.environ.get("CFG_MODEL_MAX_RSS", "").strip()
    if raw:
        absolute = _parse_size(raw)
        if absolute is None:
            return None, f"CFG_MODEL_MAX_RSS={raw!r} is not a size, ceiling disabled"
        if absolute <= 0:
            return None, "CFG_MODEL_MAX_RSS=0"
        return absolute, f"CFG_MODEL_MAX_RSS={raw}"

    try:
        fraction = float(os.environ.get("CFG_MODEL_MAX_RSS_FRACTION", DEFAULT_MAX_RSS_FRACTION))
    except ValueError:
        fraction = DEFAULT_MAX_RSS_FRACTION
    if fraction <= 0:
        return None, "CFG_MODEL_MAX_RSS_FRACTION=0"

    limit, source = memory_limit()
    if limit is None:
        return None, f"no memory limit found ({source})"
    return int(limit * fraction), f"{fraction:g} of {limit / 1048576:.0f}MB from {source}"
