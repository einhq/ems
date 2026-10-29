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
    "Like and subscribe!",
    "Textning.nu",
    "Svensktextning.nu",
    "Undertexter från Amara.org-gemenskapen",
    "Teksting av Nicolai Winther",
    "Subtitles by the Amara.org community",
])
def test_invented_lines_are_dropped(line):
    assert is_credit_line(line)


@pytest.mark.parametrize("line", [
    "Tack för att du kom!",
    "Tack för att du tittade förbi, vi ses på lördag.",
    "Nu har vi tittat på den här videon tre gånger.",
    "Vi ska titta på filmen sen.",
    "Thank you for watching the kids.",
    "Hej!",
    "",
])
def test_speech_is_kept(line):
    assert not is_credit_line(line)
