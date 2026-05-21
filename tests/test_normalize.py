"""Text normalization tests for the Kokoro TTS service.

Tests validate that numbers, years, and other textual entities are properly
converted to spoken form before being passed to the TTS engine.

Reported issues (what the tests are designed to catch):
  - Issue #1: Year "1903" reads as "nineteen hundred and 3"
              Expected: something like "nineteen oh three" / "nineteen zero three"
  - Issue #2: Number "9,000" reads as "9 pause zero zero zero"
              Expected: "nine thousand"

How to run:
    docker compose -f docker-compose.kokoro-test.yml run --rm kokoro-tts-test

Environment variables that control normalization (set at test module load):
    KOKORO_TEXT_NORMALIZATION_BACKEND  auto | nemo | regex | off  (default: auto)
    KOKORO_MODEL_DIR                   path to model files (not needed for these tests)
    KOKORO_PRONUNCIATIONS_PATH         path to pronunciations JSON (not needed)
"""
from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Configure environment BEFORE importing app.main so lru_cache'd globals pick
# up the right paths (no real model files needed for normalization-only tests).
# ---------------------------------------------------------------------------
os.environ.setdefault("KOKORO_MODEL_DIR", "/tmp/kokoro-test-models")
os.environ.setdefault("KOKORO_PRONUNCIATIONS_PATH", "/tmp/kokoro-test-pronunciations.json")

# Force auto backend (NeMo if available, regex fallback).  Individual test
# classes override this per-test where needed.
os.environ.setdefault("KOKORO_TEXT_NORMALIZATION_BACKEND", "auto")

from app.main import _normalize_text, _normalize_with_nemo  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nemo_available() -> bool:
    """Return True if NeMo normalizer initialised successfully."""
    try:
        result = _normalize_with_nemo("test")
        return result is not None
    except Exception:
        return False


requires_nemo = pytest.mark.skipif(
    not _nemo_available(),
    reason="NeMo text-processing normalizer is not available in this environment",
)


# ---------------------------------------------------------------------------
# 1. Sanity – NeMo availability
# ---------------------------------------------------------------------------


class TestNeMoAvailability:
    """Basic sanity checks that NeMo is installed and functional."""

    def test_nemo_can_be_imported(self):
        """nemo_text_processing package must be importable."""
        import nemo_text_processing  # noqa: F401

    def test_nemo_normalizer_initialises(self):
        """NeMo Normalizer must initialise without raising an exception."""
        from nemo_text_processing.text_normalization.normalize import Normalizer

        n = Normalizer(input_case="cased", lang="en")
        assert n is not None

    @requires_nemo
    def test_nemo_passthrough_plain_text(self):
        """Plain text without numbers should pass through NeMo unchanged."""
        result = _normalize_with_nemo("Hello world")
        assert result is not None
        assert "hello" in result.lower() or "Hello" in result


# ---------------------------------------------------------------------------
# 2. Year normalization  (Issue #1)
# ---------------------------------------------------------------------------


class TestYearNormalization:
    """Years should be read in the two-part spoken form, not as cardinals.

    e.g.  1903 → "nineteen oh three"   (NOT "one thousand nine hundred three")
          1776 → "seventeen seventy six"
          1950 → "nineteen fifty"
    """

    @requires_nemo
    def test_year_1903_in_sentence(self):
        """1903 inside a sentence should not be read as a plain cardinal."""
        text = "The Wright brothers made their first flight in 1903."
        result = _normalize_text(text, lang="en-us")
        result_lower = result.lower()

        print(f"\n  Input : {text!r}")
        print(f"  Output: {result!r}")

        # Cardinal form "one thousand nine hundred" must NOT appear
        assert "one thousand nine hundred" not in result_lower, (
            f"Year 1903 was normalised as a plain cardinal.\n"
            f"  Got:      {result!r}\n"
            f"  Expected: a year-form like 'nineteen oh three'"
        )

        # The year must include "nineteen"
        assert "nineteen" in result_lower, (
            f"Year 1903 should start with 'nineteen'.\n"
            f"  Got: {result!r}"
        )

    @requires_nemo
    def test_year_1903_standalone(self):
        """Standalone '1903' should be normalised to a spoken year form."""
        result = _normalize_text("1903", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Input : '1903'")
        print(f"  Output: {result!r}")

        # Must NOT be a plain cardinal
        assert "one thousand nine hundred" not in result_lower, (
            f"Standalone year 1903 was normalised as cardinal.\n"
            f"  Got:      {result!r}\n"
            f"  Expected: year form e.g. 'nineteen oh three'"
        )

        # 'nineteen' must be present
        assert "nineteen" in result_lower, (
            f"Expected 'nineteen' in year normalisation of 1903.\n"
            f"  Got: {result!r}"
        )

        # 'three' (or 'oh three' / 'zero three') must be present
        assert any(
            phrase in result_lower for phrase in ("three", "oh three", "o three", "zero three")
        ), (
            f"Expected 'three' or 'oh three' in year normalisation of 1903.\n"
            f"  Got: {result!r}"
        )

    @requires_nemo
    def test_year_1776(self):
        """1776 should be read as 'seventeen seventy six'."""
        result = _normalize_text("The Declaration of Independence was signed in 1776.", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '...1776.': {result!r}")

        assert "seventeen" in result_lower, (
            f"Expected 'seventeen' for year 1776.\n  Got: {result!r}"
        )
        assert "seventy" in result_lower or "six" in result_lower, (
            f"Expected 'seventy six' for year 1776.\n  Got: {result!r}"
        )

    @requires_nemo
    def test_year_1900(self):
        """1900 should be read as 'nineteen hundred'."""
        result = _normalize_text("1900", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '1900': {result!r}")

        assert "nineteen hundred" in result_lower or "nineteen" in result_lower, (
            f"Expected 'nineteen hundred' for year 1900.\n  Got: {result!r}"
        )

    @requires_nemo
    def test_year_1950(self):
        """1950 should be read as 'nineteen fifty'."""
        result = _normalize_text("Born in 1950.", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '...1950.': {result!r}")

        assert "nineteen" in result_lower and "fifty" in result_lower, (
            f"Expected 'nineteen fifty' for year 1950.\n  Got: {result!r}"
        )

    @requires_nemo
    def test_year_2000(self):
        """2000 should be read as 'two thousand'."""
        result = _normalize_text("The year 2000 bug.", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '...2000...': {result!r}")

        assert "two thousand" in result_lower, (
            f"Expected 'two thousand' for year 2000.\n  Got: {result!r}"
        )

    @requires_nemo
    def test_year_2024(self):
        """2024 should be read in a recognisable spoken form."""
        result = _normalize_text("In 2024, AI is everywhere.", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '...2024...': {result!r}")

        # Accept either "twenty twenty four" or "two thousand twenty four"
        ok = (
            ("twenty" in result_lower and "four" in result_lower)
            or "two thousand" in result_lower
        )
        assert ok, (
            f"Expected 'twenty twenty four' or 'two thousand twenty four' for 2024.\n"
            f"  Got: {result!r}"
        )


# ---------------------------------------------------------------------------
# 3. Comma-separated number normalization  (Issue #2)
# ---------------------------------------------------------------------------


class TestCommaNumberNormalization:
    """Comma-separated numbers should be read as whole numbers.

    e.g.  9,000    → "nine thousand"       (NOT "nine comma zero zero zero")
          10,000   → "ten thousand"
          1,000,000 → "one million"
    """

    @requires_nemo
    def test_9000_standalone(self):
        """9,000 on its own should normalise to 'nine thousand'."""
        result = _normalize_text("9,000", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Input : '9,000'")
        print(f"  Output: {result!r}")

        assert "nine thousand" in result_lower, (
            f"Expected 'nine thousand' for '9,000'.\n"
            f"  Got: {result!r}\n"
            f"  (Comma is likely not being treated as a thousands separator)"
        )

    @requires_nemo
    def test_9000_in_sentence(self):
        """9,000 inside a sentence should normalise to 'nine thousand'."""
        text = "There were 9,000 soldiers at the battle."
        result = _normalize_text(text, lang="en-us")
        result_lower = result.lower()

        print(f"\n  Input : {text!r}")
        print(f"  Output: {result!r}")

        assert "nine thousand" in result_lower, (
            f"Expected 'nine thousand' in: {result!r}"
        )

    @requires_nemo
    def test_10000(self):
        """10,000 should normalise to 'ten thousand'."""
        result = _normalize_text("10,000", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '10,000': {result!r}")

        assert "ten thousand" in result_lower, (
            f"Expected 'ten thousand' for '10,000'.\n  Got: {result!r}"
        )

    @requires_nemo
    def test_1000000(self):
        """1,000,000 should normalise to 'one million'."""
        result = _normalize_text("1,000,000", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '1,000,000': {result!r}")

        assert "one million" in result_lower, (
            f"Expected 'one million' for '1,000,000'.\n  Got: {result!r}"
        )

    @requires_nemo
    def test_1500(self):
        """1,500 should normalise to 'one thousand five hundred' or 'fifteen hundred'."""
        result = _normalize_text("1,500", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Output for '1,500': {result!r}")

        ok = "fifteen hundred" in result_lower or "one thousand five hundred" in result_lower
        assert ok, (
            f"Expected 'fifteen hundred' or 'one thousand five hundred' for '1,500'.\n"
            f"  Got: {result!r}"
        )

    @requires_nemo
    def test_comma_does_not_produce_pause_or_literal_digits(self):
        """Comma in a number must NOT result in a spoken pause or individual digits."""
        result = _normalize_text("9,000", lang="en-us")
        result_lower = result.lower()

        print(f"\n  Checking that '9,000' does not produce digit-by-digit reading: {result!r}")

        # Should NOT contain any of these bad patterns
        for bad in ("pause", "zero zero zero", "0 0 0", "comma"):
            assert bad not in result_lower, (
                f"Bad pattern {bad!r} found in normalised output.\n"
                f"  Got: {result!r}\n"
                f"  (The comma in '9,000' is not being treated as thousands separator)"
            )


# ---------------------------------------------------------------------------
# 4. General number normalization
# ---------------------------------------------------------------------------


class TestGeneralNumberNormalization:
    """Basic number normalization for completeness."""

    @requires_nemo
    def test_single_digit(self):
        result = _normalize_text("I have 5 cats.", lang="en-us")
        assert "five" in result.lower(), f"Expected 'five': {result!r}"

    @requires_nemo
    def test_hundred(self):
        result = _normalize_text("100", lang="en-us")
        assert "hundred" in result.lower(), f"Expected 'hundred': {result!r}"

    @requires_nemo
    def test_thousand_without_comma(self):
        result = _normalize_text("1000", lang="en-us")
        result_lower = result.lower()
        assert "thousand" in result_lower or "one thousand" in result_lower, (
            f"Expected 'thousand' for 1000: {result!r}"
        )

    def test_decimal_regex_fallback(self):
        """Regex backend should replace decimal points with 'point'."""
        original = os.environ.get("KOKORO_TEXT_NORMALIZATION_BACKEND")
        try:
            os.environ["KOKORO_TEXT_NORMALIZATION_BACKEND"] = "regex"
            result = _normalize_text("3.14", lang="en-us")
            assert "point" in result.lower(), (
                f"Expected 'point' from regex fallback for '3.14': {result!r}"
            )
        finally:
            if original is None:
                os.environ.pop("KOKORO_TEXT_NORMALIZATION_BACKEND", None)
            else:
                os.environ["KOKORO_TEXT_NORMALIZATION_BACKEND"] = original

    def test_off_backend_passthrough(self):
        """'off' backend should return text unchanged (after pronunciation substitution)."""
        original = os.environ.get("KOKORO_TEXT_NORMALIZATION_BACKEND")
        try:
            os.environ["KOKORO_TEXT_NORMALIZATION_BACKEND"] = "off"
            result = _normalize_text("9,000", lang="en-us")
            assert "9,000" in result, (
                f"Expected '9,000' unchanged with 'off' backend: {result!r}"
            )
        finally:
            if original is None:
                os.environ.pop("KOKORO_TEXT_NORMALIZATION_BACKEND", None)
            else:
                os.environ["KOKORO_TEXT_NORMALIZATION_BACKEND"] = original

    def test_non_english_passthrough(self):
        """Non-English text should bypass NeMo normalization entirely."""
        result = _normalize_text("1903", lang="zh")
        assert "1903" in result, (
            f"Expected '1903' unchanged for non-English lang='zh': {result!r}"
        )


# ---------------------------------------------------------------------------
# 5. Diagnostic – current NeMo behaviour (always runs, never fails)
#    Run with -s to see the printed output:  pytest tests/ -s -v
# ---------------------------------------------------------------------------


class TestCurrentNeMoBehaviorDiagnostic:
    """Diagnostic tests that print current NeMo output without asserting.

    These tests ALWAYS PASS but print what NeMo currently produces so you
    can see exactly what needs to be fixed.  Run with -s to capture stdout:

        pytest tests/ -s -v -k "Diagnostic"
    """

    inputs = [
        # (label, text, lang)
        ("year_1903_sentence", "The Wright brothers flew in 1903.", "en-us"),
        ("year_1903_standalone", "1903", "en-us"),
        ("year_1776", "America declared independence in 1776.", "en-us"),
        ("year_1950", "Born in 1950.", "en-us"),
        ("year_2000", "The year 2000 was memorable.", "en-us"),
        ("year_2024", "In 2024, AI is everywhere.", "en-us"),
        ("comma_9000", "9,000", "en-us"),
        ("comma_9000_sentence", "There are 9,000 soldiers.", "en-us"),
        ("comma_10000", "10,000", "en-us"),
        ("comma_1000000", "1,000,000", "en-us"),
        ("comma_1500", "1,500", "en-us"),
        ("cardinal_100", "100", "en-us"),
        ("cardinal_1000", "1000", "en-us"),
        ("decimal_3_14", "3.14", "en-us"),
    ]

    @pytest.mark.parametrize("label,text,lang", inputs)
    def test_print_nemo_output(self, label, text, lang):
        """Print current NeMo normalisation output for the given input."""
        nemo_out = _normalize_with_nemo(text)
        full_out = _normalize_text(text, lang=lang)

        print(f"\n  [{label}]")
        print(f"    input        : {text!r}")
        print(f"    nemo direct  : {nemo_out!r}")
        print(f"    _normalize_text: {full_out!r}")
        # Never fail — this is purely diagnostic
        assert True
