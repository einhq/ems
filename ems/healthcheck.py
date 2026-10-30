"""`python3 -m ems.healthcheck` — the container's health check: exit 0 when `GET /v0/health` answers
ok and the batch model worker is running.

`ok` in health says only that the front answers, which is what a caller asks before it relies on
the answer's other fields. The container is healthy when the work can be done too: a front whose
batch worker is down (between the supervisor's restarts, or never started) answers every model
op `model_unavailable`, and `podman ps` should say so.

Run by podman inside the container. `make run` passes it as `--health-cmd` rather than the image
declaring a HEALTHCHECK, which podman drops when it builds the OCI format. It asks the local port
with the container's own key; with no key configured authentication is off and none is sent.

Podman gives it 60 s and 5 tries, so unhealthy means EMS has not answered for minutes. A host
busy with a crawl can stall every container's check at once, the database's `psql` as well as
this one, on starting the process rather than on the answer (fry, 2026-10-30): a tighter bound
then marks unhealthy an EMS that serves every request.
"""

from __future__ import annotations

import json
import sys
import urllib.request

from .config import Settings

# Longer than what health itself may wait (`nvidia-smi` twice) on a loaded host, and shorter than
# podman's 60 s, so a slow answer still says why.
TIMEOUT_SECONDS = 45


def main() -> int:
    settings = Settings.from_env()
    request = urllib.request.Request(f"http://127.0.0.1:{settings.port}/v0/health")
    if settings.keys:
        request.add_header("Authorization", f"Bearer {next(iter(settings.keys))}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.load(response)
    except (OSError, ValueError) as exc:
        print(f"unhealthy: {exc}", file=sys.stderr)
        return 1
    if not body.get("ok"):
        print("unhealthy: health is not ok", file=sys.stderr)
        return 1
    if not (body.get("worker") or {}).get("running"):
        print("unhealthy: the batch model worker is not running", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
