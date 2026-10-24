"""The one error shape.

Every failure a caller sees is `{ok: false, error: {code, message, permanent}}` with the HTTP
status the code implies. `permanent` is the part worth getting right: it is what tells a caller
with a retry queue whether to try again. A truncated container, an unsupported format and a
path outside the roots will fail the same way forever; a full queue, a loading model and a
missed deadline will not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Code:
    name: str
    status: int
    permanent: bool | None  # None: the raiser decides


UNAUTHORIZED = Code("unauthorized", 401, None)
INVALID_REQUEST = Code("invalid_request", 400, True)
PATH_OUTSIDE_ROOTS = Code("path_outside_roots", 403, True)
NOT_FOUND = Code("not_found", 404, True)
UNSUPPORTED = Code("unsupported", 415, True)
UNDECODABLE = Code("undecodable", 422, True)
REFUSED = Code("refused", 422, True)
BUSY = Code("busy", 503, False)
DEADLINE_EXCEEDED = Code("deadline_exceeded", 504, False)
MODEL_UNAVAILABLE = Code("model_unavailable", 503, False)
TOOL_FAILED = Code("tool_failed", 500, False)
INTERNAL = Code("internal", 500, False)


class EmsError(Exception):
    """A failure with a code. Raise it anywhere; the app turns it into the envelope."""

    def __init__(self, code: Code, message: str, *, permanent: bool | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.permanent = code.permanent if permanent is None else permanent
        self.retry_after = retry_after

    def envelope(self) -> dict:
        error: dict = {"code": self.code.name, "message": self.message}
        if self.permanent is not None:
            error["permanent"] = self.permanent
        return {"ok": False, "error": error}


def invalid(message: str) -> EmsError:
    return EmsError(INVALID_REQUEST, message)
