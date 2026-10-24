"""Where EMS may read and write.

EMS is a network service that opens whatever path it is handed, so the roots are the security
boundary: a path is accepted only if, after every symlink is followed, it still lies under a
configured root — and for an output, under one mounted read-write. A symlink that leaves the
roots is refused even though its name is inside them, because what would be read is outside.

`EMS_ROOTS` is `path:mode,path:mode,…`, e.g. `/files:ro,/old:ro,/files-volatile:rw`. An empty
setting means EMS may touch nothing, which makes a misconfigured deployment fail on its first
request instead of quietly serving the whole disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import EmsError, NOT_FOUND, PATH_OUTSIDE_ROOTS, invalid


@dataclass(frozen=True)
class Root:
    path: str
    writable: bool

    def contains(self, resolved: str) -> bool:
        return resolved == self.path or resolved.startswith(self.path + os.sep)


def parse_roots(spec: str) -> tuple[Root, ...]:
    roots: list[Root] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        path, _, mode = entry.partition(":")
        mode = mode or "ro"
        if mode not in ("ro", "rw"):
            raise ValueError(f"root {entry!r}: mode must be ro or rw")
        if not os.path.isabs(path):
            raise ValueError(f"root {entry!r}: path must be absolute")
        roots.append(Root(os.path.normpath(path), mode == "rw"))
    return tuple(roots)


class Roots:
    def __init__(self, roots: tuple[Root, ...]):
        self.roots = roots

    def describe(self) -> list[dict]:
        return [{"path": r.path, "mode": "rw" if r.writable else "ro"} for r in self.roots]

    def _under(self, resolved: str, *, writable: bool) -> Root | None:
        for root in self.roots:
            if root.contains(resolved) and (root.writable or not writable):
                return root
        return None

    def input(self, path: str) -> str:
        """Resolve an input path and refuse anything not under a readable root.

        Returns the real path, so a symlink into another root (an original under `/files`
        pointing into `/old`) is opened where the bytes are; both roots must be configured.
        """
        if not isinstance(path, str) or not path:
            raise invalid("file.path is required")
        if not os.path.isabs(path):
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not an absolute path")
        resolved = os.path.realpath(path)
        if self._under(resolved, writable=False) is None:
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: outside the configured roots")
        if not os.path.isfile(resolved):
            raise EmsError(NOT_FOUND, f"{path}: no such file")
        return resolved

    def output_dir(self, path: str, *, create: bool = True) -> str:
        """Resolve a directory outputs will be written into; under a writable root, created if asked."""
        if not isinstance(path, str) or not path:
            raise invalid("output.dir is required")
        if not os.path.isabs(path):
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not an absolute path")
        real = os.path.realpath(path)
        if self._under(real, writable=True) is None:
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: outside the writable roots")
        if not os.path.isdir(real):
            if not create:
                raise EmsError(NOT_FOUND, f"{path}: output directory does not exist")
            os.makedirs(real, exist_ok=True)
        return real

    def output(self, path: str) -> str:
        """Resolve an output path: its directory must exist under a writable root.

        The file itself need not exist. The *directory* is resolved rather than the file, so a
        dangling symlink at the output path cannot redirect the write; the final component is
        appended to the real directory and must be a plain name.
        """
        if not isinstance(path, str) or not path:
            raise invalid("output.path is required")
        if not os.path.isabs(path):
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not an absolute path")
        directory, name = os.path.split(os.path.normpath(path))
        if not name or name in (".", ".."):
            raise invalid(f"{path}: output must name a file")
        real_dir = os.path.realpath(directory)
        if self._under(real_dir, writable=True) is None:
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: outside the writable roots")
        if not os.path.isdir(real_dir):
            raise EmsError(NOT_FOUND, f"{path}: output directory does not exist")
        target = os.path.join(real_dir, name)
        if os.path.islink(target):
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: output path is a symlink")
        return target
