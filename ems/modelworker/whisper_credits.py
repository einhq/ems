"""The lines whisper invents over audio that holds no speech.

Whisper learned from subtitled video, and subtitles end in a thank-you for watching or a credit
to whoever made them. Over room noise, wind or footsteps the decoder writes one of those, fluent
and confident, often as the only line of a clip and timed across all of it. The no-speech ceiling
(model_registry.WHISPER_NO_SPEECH_PROB_MAX) catches most of them but not all, so the rest are
dropped by what they say.

A thank-you line is dropped only when it is the whole segment: someone who says it in passing
keeps it. A credit is dropped wherever it stands, because nobody says "Textning.nu" aloud.
"""

from __future__ import annotations

import re

# Whole-segment lines, matched after `_normalize`.
_THANKS = [
    # Swedish
    r"(tusen )?tack för att (du|ni) (har )?(tittat|tittade|såg|sett|lyssnat|lyssnade)"
    r"( på (den här|denna|min|vår) (video|videon|videoklipp|film|filmen|kanal|kanalen))?",
    # Norwegian
    r"(tusen )?takk for at (du|dere) (har )?(så|sett|ser|lyttet|hørte)( på)?( (denne|videoen|videoen min))?",
    # Danish
    r"(tusind )?tak (fordi|for at) (du|i) (har )?(så|set|ser|lyttede)( med)?",
    # English
    r"thank you (so much )?for watching",
    r"thanks (so much )?for watching",
    r"(please )?(like and )?subscribe( to (my|the|our) channel)?",
]

# Credit markers, anywhere in a segment.
_CREDITS = [
    r"amara ?org",
    r"textning ?nu",
    r"undertexter från",
    r"undertextning av",
    r"teksting av",
    r"tekstet av",
    r"danske tekster",
    r"subtitles by",
]

_THANKS_RE = re.compile("|".join(f"(?:{p})" for p in _THANKS))
_CREDITS_RE = re.compile("|".join(f"(?:{p})" for p in _CREDITS))


def _normalize(text: str) -> str:
    """Lower case, punctuation as space, single spaces: `Tack för att du tittade!` → `tack för att du tittade`."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def is_credit_line(text: str) -> bool:
    """Whether `text`, one decoded segment, is a line whisper invents rather than one it heard."""
    line = _normalize(text)
    return bool(_THANKS_RE.fullmatch(line) or _CREDITS_RE.search(line))
