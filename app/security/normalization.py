"""Unicode normalization + language identification (Governance & Guardrails §6.3).

The cheapest, highest-value multilingual defense runs *before* any pattern check:
NFKC-normalize, strip zero-width characters, and fold common homoglyphs so a
"multilingual"/Unicode-obfuscated jailbreak can't walk past an English blocklist.
A coarse language/script guess then drives the per-language policy (wiring is a
later phase). Pure + keyless.
"""

from __future__ import annotations

import unicodedata

# Zero-width + invisible formatting characters used to break up trigger words.
_ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E, 0x200E, 0x200F], None
)

# Common homoglyphs (Cyrillic/Greek lookalikes) folded to their Latin twin.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
        "і": "i", "ѕ": "s", "ԁ": "d", "ո": "n", "Ⅼ": "L",
        "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
        "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "ο": "o",
    }
)


def normalize(text: str) -> str:
    """NFKC-normalize, strip zero-width chars, and fold homoglyphs to Latin."""
    if not text:
        return text
    text = text.translate(_ZERO_WIDTH)
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_HOMOGLYPHS)


def normalize_tagged(text: str) -> tuple[str, list[str]]:
    """Normalize and report whether anything changed (``guardrail:normalized``)."""
    clean = normalize(text)
    return clean, (["guardrail:normalized"] if clean != text else [])


def dominant_script(text: str) -> str:
    """The most common Unicode script in ``text`` (latin | cyrillic | cjk | …)."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        script = next(
            (s for s in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "HIRAGANA",
                         "KATAKANA", "HANGUL", "CJK") if s in name),
            "OTHER",
        )
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=counts.get).lower()


def guess_language(text: str) -> str:
    """Coarse, keyless language guess from the dominant script (default ``en``)."""
    script = dominant_script(text)
    return {
        "latin": "en", "cyrillic": "ru", "greek": "el", "arabic": "ar",
        "hebrew": "he", "hiragana": "ja", "katakana": "ja", "hangul": "ko",
        "cjk": "zh", "unknown": "unknown",
    }.get(script, "unknown")
