"""
Mock HS-classification table (UC3) — realistic, attributed, easy to find/replace.

Each row maps goods keywords → a Harmonized System candidate. The
``MockClassificationAdapter`` substring-matches a description against these
keywords and returns the matches ranked by confidence, with an explicit
low-confidence fallback for the unknown case. Not authoritative — a demo corpus.
"""

from __future__ import annotations

# (match_keywords, hs_code, title, confidence)
HS_TABLE: list[tuple[tuple[str, ...], str, str, float]] = [
    (("power bank", "powerbank", "power-bank"), "8507.60", "Lithium-ion accumulators", 0.93),
    (("laptop", "notebook computer"), "8471.30", "Portable data-processing machines", 0.92),
    (("smartphone", "mobile phone", "cell phone"), "8517.13", "Smartphones", 0.92),
    (("headphone", "earphone", "earbud"), "8518.30", "Headphones and earphones", 0.90),
    (("camera",), "8525.89", "Television cameras / digital cameras", 0.84),
    (("drone", "uav", "quadcopter"), "8806", "Unmanned aircraft", 0.88),
    (("led",), "8539.50", "LED lamps", 0.82),
    (("t-shirt", "tee", "cotton shirt"), "6109.10", "T-shirts, cotton, knitted", 0.90),
    (("handbag", "purse"), "4202.21", "Handbags with outer surface of leather", 0.88),
    (("shoe", "footwear", "sneaker"), "6403.99", "Footwear with leather uppers", 0.86),
    (("sunglasses",), "9004.10", "Sunglasses", 0.87),
    (("watch", "wristwatch"), "9102", "Wrist-watches", 0.85),
    (("coffee", "roasted coffee"), "0901.21", "Coffee, roasted, not decaffeinated", 0.89),
    (("wine",), "2204", "Wine of fresh grapes", 0.88),
    (("perfume", "fragrance"), "3303", "Perfumes and toilet waters", 0.86),
    (("cosmetic", "makeup", "skincare"), "3304", "Beauty / make-up preparations", 0.85),
    (("toy", "toys"), "9503", "Toys", 0.84),
    (("book", "books"), "4901", "Printed books and brochures", 0.90),
    (("bicycle", "bike"), "8712", "Bicycles, non-motorised", 0.87),
    (("supplement", "vitamin"), "2106.90", "Food preparations (supplements)", 0.80),
]

# Explicit fallback so the unknown case is honest, never a fabricated match.
HS_FALLBACK: tuple[str, str, float] = ("9999.99", "Unclassified goods (manual review)", 0.20)
