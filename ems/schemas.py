"""Request shapes shared by the ops. See docs/api.md §3."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Options(Strict):
    priority: Literal["interactive", "batch"] = "batch"
    deadline: float | None = None

    @property
    def interactive(self) -> bool:
        return self.priority == "interactive"


class FileRef(Strict):
    path: str
    angle: Literal[0, 90, 180, 270] = 0
    mirror: bool = False
    mimetype: str | None = None


class Box(Strict):
    """Fractions of the display frame, origin top-left."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class OutputSpec(Strict):
    path: str
    format: str | None = None
    quality: int | None = Field(default=None, ge=1, le=100)
    sample_rate: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)


class WithOptions(Strict):
    options: Options = Field(default_factory=Options)
