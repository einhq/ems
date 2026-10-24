"""The language a picture's text is written in, from the lines the text reader kept.

lingua (high accuracy, every language it knows) on the CPU: ~160 MB and a millisecond or two a
reading. Its confidences are relative — the top language's share among all — so it answers low
when the text could be several languages, which an OCR reading of a name or a date always could;
there the answer is no language rather than a guess (ein/core#670).
"""

from __future__ import annotations

import os

# Fewer letters than this is no language: a name, a date, a house number.
MIN_LETTERS = int(os.environ.get('CFG_OCR_LANGUAGE_MIN_LETTERS', '40'))
# Below this share the reading could be several languages.
MIN_CONFIDENCE = float(os.environ.get('CFG_OCR_LANGUAGE_MIN_CONFIDENCE', '0.5'))


class LanguageIdentifier:
    """Text in, `{"language": "<ISO 639-1>", "confidence": <0..1>}` out, or None when it cannot say."""

    model = "lingua"

    def __init__(self):
        from lingua import LanguageDetectorBuilder

        self.detector = LanguageDetectorBuilder.from_all_languages().build()

    def detect(self, text: str) -> dict | None:
        if sum(c.isalpha() for c in text) < MIN_LETTERS:
            return None
        values = self.detector.compute_language_confidence_values(text)
        if not values or values[0].value < MIN_CONFIDENCE:
            return None
        top = values[0]
        return {"language": top.language.iso_code_639_1.name.lower(), "confidence": round(float(top.value), 3)}
