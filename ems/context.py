"""What every op needs a handle on: settings, roots, admission, the worker, the tools."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

from .admission import Admission
from .card import CardArbiter
from .config import Settings
from .roots import Roots
from .tools.audio import AudioCache
from .tools.exiftool import ExifTool
from .worker import Worker


@dataclass
class Context:
    settings: Settings
    roots: Roots
    admission: Admission
    worker: Worker
    exiftool: ExifTool | None
    audio: AudioCache
    started_at: float = field(default_factory=time.time)
    warnings: list[str] = field(default_factory=list)
    # The worker search queries go to, or None to send them to `worker` (EMS_QUERY_WORKER=0).
    query_worker: Worker | None = None
    # Who has the card (ems/card.py); off without a card or a query worker.
    card: CardArbiter | None = None

    def for_queries(self, family: str) -> Worker:
        """Where a search query for `family` is embedded: the card worker while it holds the family
        (ems/card.py), else the query worker while it is up, else the batch worker.

        The query worker holds only the query families (QUERY_FAMILIES) on the CPU, at normal
        priority, while the batch worker runs niced, so a query does not share the CPU and the
        interpreter lock with batch work. Falling back keeps search answering — slowly — through
        a query worker restart.
        """
        card = self.card.for_queries(family) if self.card is not None else None
        if card is not None:
            return card
        query = self.query_worker
        return query if query is not None and query.running else self.worker

    def scratch_dir(self, prefix: str) -> str:
        os.makedirs(self.settings.scratch, exist_ok=True)
        return tempfile.mkdtemp(prefix=f"{prefix}-", dir=self.settings.scratch)
