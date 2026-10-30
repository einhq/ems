"""`text.embed`."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import Field, model_validator

from ..context import Context
from ..schemas import WithOptions
from ..streaming import Outcome, Progress, run_op
from ._models import model_slot


class EmbedRequest(WithOptions):
    # One text, answered with `embedding`, or several, answered with `embeddings` in order.
    text: str | None = None
    texts: list[str] | None = Field(default=None, min_length=1, max_length=256)
    # The configured embedder (capabilities `models.text_embed`) is the only one. Naming it is
    # allowed; any other name is refused rather than answered from another space.
    model: Literal["current"] = "current"
    # A search query and the text it should find are embedded differently.
    role: Literal["query", "document"] = "document"

    @model_validator(mode="after")
    def _one_input(self) -> "EmbedRequest":
        if (self.text is None) == (self.texts is None):
            raise ValueError("give exactly one of text or texts")
        return self


async def embed(ctx: Context, req: EmbedRequest, progress: Progress) -> Outcome:
    texts = req.texts if req.texts is not None else [req.text or ""]
    # A search query goes to the query family (Context.for_queries), and so do documents someone
    # is waiting on (`priority: interactive`, a hand edit being saved), which would otherwise
    # queue behind a crawl's; batch documents go to the batch worker.
    quick = req.role == "query" or req.options.interactive
    async with model_slot(ctx, req.options):
        worker = ctx.for_queries("text-query") if quick else ctx.worker
        payload: dict = {"texts": texts, "role": req.role}
        if quick and req.role == "document":
            payload["family"] = "query"
        device = ctx.card.document_device() if ctx.card is not None and not quick else None
        if device is not None:
            payload["device"] = device
        raw = await worker.call("embed_texts", payload, deadline=req.options.deadline)
    vectors = [list(vector) for vector in raw.get("embeddings") or []]
    producer = str(raw.get("model") or "text-embed")
    dimension = len(vectors[0]) if vectors else 0
    if req.texts is None:
        return Outcome({"embedding": vectors[0], "dimension": dimension}, producer=producer)
    return Outcome({"embeddings": vectors, "dimension": dimension}, producer=producer)


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/text/embed")
    async def text_embed(request: Request, body: EmbedRequest):
        return await run_op(request, lambda progress: embed(ctx, body, progress))
