"""Known visible label text, not verified product identities or dosages.

Map canonical display names to alternative legitimate spellings/languages.
Do not add OCR mistakes as aliases: those should be evaluated by fuzzy matching.
"""

LABELS = {
    "Melatonina": ["melatonin"],
    "Ibuprofeno": ["ibuprofen"],
    "Colgate": [],
    "NESCAFÉ": [],
}

MIN_MATCH_SCORE = 0.75
MIN_MATCH_MARGIN = 0.15
