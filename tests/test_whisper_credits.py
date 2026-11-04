"""The thank-you and credit lines whisper invents over noise are dropped; speech that only resembles them is kept."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

from whisper_credits import is_credit_line  # noqa: E402


@pytest.mark.parametrize("line", [
    # The one cue of a silent 62-second clip on fry.
    "Tack för att du har tittat på den här videon!",
    "Tack för att du tittade!",
    "Tack för att ni tittade.",
    "Tusen tack för att du lyssnade!",
    "Takk for at du så på!",
    "Takk for at dere har sett på.",
    "Tak fordi du så med!",
    "Thank you for watching!",
    "Thanks for watching.",
    "Thank you for watching my video.",
    "Like and subscribe!",
    "Takk for at du så med.",
    "Grazie a tutti per aver guardato il video.",
    # Two lines in one segment.
    "Sampai jumpa di video selanjutnya. Terima kasih telah menonton!",
    "Textning.nu",
    "Svensktextning.nu",
    "Undertexter från Amara.org-gemenskapen",
    "Teksting av Nicolai Winther",
    "Subtitles by the Amara.org community",
    "Субтитры сделал DimaTorzok",
])
def test_invented_lines_are_dropped(line):
    assert is_credit_line(line)


@pytest.mark.parametrize("line", [
    "Tack för att du kom!",
    "Tack för att du tittade förbi, vi ses på lördag.",
    "Nu har vi tittat på den här videon tre gånger.",
    "Vi ska titta på filmen sen.",
    "Thank you for watching the kids.",
    "Thank you for watching. Now let's eat.",
    "Grazie per aver guardato i bambini.",
    "Hej!",
    "",
])
def test_speech_is_kept(line):
    assert not is_credit_line(line)
