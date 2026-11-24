"""Which way up is a scanned page? Upright, sideways, or upside down.

A NOTE ONLY: the OCR never turns a page on this module's word (vlm_ocr.render).
It did on 2026-10-10 and that did more harm than good — of 12 pages it turned in
a check of the GAD corpus, all 12 were upright: English forms whose underlines,
fill-in blanks and table rules pass for upside-down Hindi headlines, and a
narrow-column Hindi table taken for sideways text. Its opinion goes into the
sidecar ("turn_suggested") for a person to look at: scripts/turn_review.py.

Why it exists (scripts/ocr_hindi_eval.py, 2026-10-10): an upside-down GAD scan made
the OCR model write fluent Hindi that is not on the page — "अधिसूचना संख्या: 25
स्वामी नारायण दास जी…" — and then loop. The page gate cannot catch that.

How, without a model: Devanagari hangs every word from a headline, the
शिरोरेखा — the densest pixel row of a text line, near its TOP (matras above it
are small; the letter bodies hang below). Upside down, that dense row sits near
the BOTTOM of the line. Sideways, the text lines run vertically: the page has
gaps between columns of ink instead of between rows.

    orientation(gray) -> (degrees, reason)    degrees = clockwise turn that makes
                                              the page upright: 0, 90, 180 or 270

Conservative by design: anything unclear is 0 (leave the page as it is). A page
turned wrongly is a page lost, so a turn needs a clear majority of many lines.
English-only pages have no headline: the 180° test abstains on them (0), the
sideways test still works.
"""

from __future__ import annotations

import numpy as np

INK_LEVEL = 150            # grey < this is ink (0 = black)
MIN_INK_SHARE = 0.004      # less ink than this: nothing to judge
GAP_DENSITY = 0.01         # a row/column with < 1% ink pixels is a gap between lines
MIN_LINE_PX, MAX_LINE_PX = 6, 90   # a text line's height at ~110 dpi
HEADLINE_PEAK = 2.0        # headline row ink >= this x the line's median row ink
HEADLINE_RUN = 0.6         # its ink runs average >= this x the line height: one bar per word.
                           # English letters' feet (serifs, baselines) are dense rows too, but in
                           # short runs (0.2-0.3x); without this, upright English read as
                           # upside-down Hindi (measured: 14 of 59 CHiPS pages)
MIN_TURN_VOTES = 8         # lines that must agree before a page is turned …
TURN_MARGIN = 3.0          # … and outvote every other way up by this factor


def _gap_share(ink: np.ndarray, axis: int) -> float:
    """Share of gap rows (axis=1) or gap columns (axis=0) inside the inked block.

    Measured as ink DENSITY, not relative to the densest row: one dense stamp
    border or table rule would otherwise make every normal row look like a gap.
    """
    prof = ink.sum(axis)
    nz = np.nonzero(prof)[0]
    if len(nz) < 10:
        return 0.0
    other = ink.shape[axis]
    return float((prof[nz[0]:nz[-1] + 1] < GAP_DENSITY * other).mean())


def _lines(ink: np.ndarray) -> list[tuple[int, int]]:
    rows = ink.sum(1)
    on = rows >= max(1.0, GAP_DENSITY * ink.shape[1])
    out, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(on)))
    return [(a, b) for a, b in out if MIN_LINE_PX <= b - a <= MAX_LINE_PX]


def headline_votes(ink: np.ndarray) -> tuple[int, int]:
    """(lines whose headline is near the top, lines whose headline is near the bottom).

    Only lines that HAVE a Devanagari headline vote: a row much denser than the rest
    of the line, made of word-long bars — several of them (one bar across the whole
    line is an underline or a table rule, not a headline).
    """
    up = down = 0
    for a, b in _lines(ink):
        band = ink[a:b]
        prof = band.sum(1).astype(float)
        med = float(np.median(prof))
        p = int(prof.argmax())
        if med <= 0 or prof[p] < HEADLINE_PEAK * med:
            continue
        xs = np.nonzero(band.any(0))[0]
        row = band[p, xs[0]:xs[-1] + 1].astype(np.int8)
        edges = np.diff(np.concatenate(([0], row, [0])))
        runs = np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]
        h = b - a
        if len(runs) < 2 or runs.mean() < HEADLINE_RUN * h or runs.max() > 0.9 * len(row):
            continue
        r = p / max(h - 1, 1)
        if r < 0.45:
            up += 1
        elif r > 0.55:
            down += 1
    return up, down


def orientation(gray: np.ndarray) -> tuple[int, str]:
    """Clockwise degrees (0/90/180/270) that turn this greyscale page upright.

    Each way up gets the votes of the text lines that read correctly that way
    (headline on top). A page is turned only when one other way up wins by a
    wide margin over every alternative, the current one included; a sideways
    turn also needs the page's gaps to run between columns, not rows.
    """
    ink = gray < INK_LEVEL
    if ink.mean() < MIN_INK_SHARE:
        return 0, "too little ink to judge"
    h_up, h_down = headline_votes(ink)
    v_up, v_down = headline_votes(np.rot90(ink, -1))   # as if turned 90° clockwise
    votes = {0: h_up, 180: h_down, 90: v_up, 270: v_down}
    best = max(votes, key=votes.get)
    rest = max(v for k, v in votes.items() if k != best)
    tally = ", ".join(f"{k}°:{v}" for k, v in votes.items())
    if best == 0 or votes[best] < MIN_TURN_VOTES or votes[best] < TURN_MARGIN * max(rest, 1):
        return 0, (f"left as is (headline votes {tally})" if any(votes.values())
                   else "no Devanagari headlines; left as is")
    if best in (90, 270):
        gr, gc = _gap_share(ink, 1), _gap_share(ink, 0)
        if gc <= gr:
            return 0, f"left as is: headlines say {best}° but the gaps run between rows (votes {tally})"
    return best, f"turned {best}°: headlines on top that way (votes {tally})"


def gray_array(pix) -> np.ndarray:
    """A PyMuPDF greyscale Pixmap -> 2-D uint8 array."""
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
