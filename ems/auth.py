"""Bearer keys. One role; a key either opens everything or nothing."""

from __future__ import annotations

from fastapi import Request

from .errors import EmsError, UNAUTHORIZED


def require_key(request: Request) -> None:
    keys: frozenset[str] = request.app.state.settings.keys
    if not keys:
        return  # authentication is off; config.load_keys says so at startup
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() not in keys:
        raise EmsError(UNAUTHORIZED, "missing or unknown key")
