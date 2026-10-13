"""Parse AveiroBus timetable PDFs into trips and their stop-to-stop calls.

AveiroBus publishes one PDF per urban line, and the timetable inside it is a
single embedded bitmap (about 3400x2200 px, some 430 dpi) rather than text:
``pdftotext`` returns nothing but the page number. Text extraction therefore
means OCR, and OCR over a whole page is exactly where timetable digits get
silently lost.

So the page is not read as prose:

1. the bitmap is lifted out of the PDF at its native resolution, with no
   resampling anywhere;
2. the *grid* is recovered from the pixels — the red header bands give the
   column groups and the service each column belongs to, the OCR word boxes
   give the rows and the column pitch;
3. each cell is then read from two independent OCR passes (different page
   segmentation, scale and binarisation) and OCR'd from a tight crop of that
   cell alone when they disagree, or when the value would break the invariant
   that a trip's times never run backwards.

Every cell that needed that third opinion is counted and reported, so the
confidence of an extraction is visible rather than assumed.
"""

from __future__ import annotations

import io
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from inmotion.bus.records import Call, Trip

TIME = re.compile(r"^(\d{1,2})[:.](\d{2})(\*?)$")
DASH = re.compile(r"^[-–—−_~\s]+$")
NOTES = re.compile(r"[*†‡]+")
WORDISH = re.compile(r"[0-9A-Za-zÀ-ÿ]")

# The timetables are Portuguese, and the language pack is not part of the
# Python dependencies, so a machine without it fails here rather than silently
# reading every caption and stop name through the wrong model.
OCR_LANGUAGE = "por"
OCR_LANGUAGE_PACKAGE = "tesseract-ocr-por"

# Header bands are a different colour on every line (red on line 1, blue on
# ``azul``, green on ``verde``), so bands are found by saturation rather than by
# matching any particular brand colour.
SATURATION = 45
BAND_FRACTION = 0.20
PURPLE = np.array([169, 145, 196])
PURPLE_TOLERANCE = 60

# Captions are matched on unaccented, lowercased text: the bitmap OCR reads
# "DIAS ÚTEIS" as both "DIAS UTEIS" and "DIAS UTEI", and the accents are not
# what makes a caption a caption.
SERVICE_HEADER = re.compile(r"dias\s+uteis|sabados?|domingos?|feriados?|fim\s+de\s+semana")

SERVICES = (
    ("fim de semana", "weekend"),
    ("dias uteis", "weekday"),
    ("sabado", "saturday"),
    ("domingo", "sunday"),
    ("feriado", "holiday"),
)


def unaccent(text: str) -> str:
    """Lowercase, unaccented, hyphen-free text, for matching captions."""
    folded = unicodedata.normalize("NFKD", text.lower().replace("-", " "))
    return "".join(c for c in folded if not unicodedata.combining(c))


TIME_WHITELIST = "0123456789:*.-–— "

# Every `tsv` render starts with this header; anything else means tesseract did
# not run the output format it was asked for.
TSV_HEADER = "level\t"


@dataclass(frozen=True)
class Word:
    """One OCR word box, in image pixels."""

    left: int
    top: int
    width: int
    height: int
    text: str
    conf: float

    @property
    def x(self) -> float:
        return self.left + self.width / 2

    @property
    def y(self) -> float:
        return self.top + self.height / 2

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass
class Band:
    """One red column-group header: the x-range it labels and its caption."""

    y0: int
    y1: int
    x0: int
    x1: int
    label: str = ""

    @property
    def service(self) -> str:
        low = unaccent(self.label)
        for needle, name in SERVICES:
            if needle in low:
                return name
        return "unknown"

    def covers(self, x: float) -> bool:
        return self.x0 - 8 <= x <= self.x1 + 8


@dataclass
class Block:
    """One timetable on the page: a stop list running against a set of trips."""

    index: int
    top: int
    bottom: int
    columns: list[float]
    rows: list[float]
    stops: list[str]
    times: list[list[str]]
    bands: list[Band] = field(default_factory=list)
    highlighted: set[int] = field(default_factory=set)
    colours: list[str] = field(default_factory=list)
    repaired: int = 0
    warnings: list[str] = field(default_factory=list)

    def service_of(self, column: int) -> str:
        """The service a column runs under.

        When several bands sit on one header row they are side by side and each
        labels the columns under it (BusWay-style). When a block has a single
        band, that band heads the whole block and its caption is narrow — the
        text-based AveiroBus timetables stack their services that way.
        """
        if len(self.bands) == 1:
            return self.bands[0].service
        for band in self.bands:
            if band.covers(self.columns[column]):
                return band.service
        return "unknown"


# -- image and OCR ---------------------------------------------------------


def timetable_image(pdf_path: Path) -> tuple[Image.Image, pymupdf.Rect]:
    """The largest embedded bitmap of the document: the timetable itself."""
    best: tuple[int, bytes, pymupdf.Rect] | None = None
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for image in page.get_images(full=True):
                payload = doc.extract_image(image[0])
                area = int(payload["width"]) * int(payload["height"])
                if best is None or area > best[0]:
                    rects = page.get_image_rects(image[0])
                    rect = rects[0] if rects else pymupdf.Rect()
                    best = (area, payload["image"], rect)
    if best is None:
        raise ValueError(f"no embedded timetable image in {pdf_path}")
    return Image.open(io.BytesIO(best[1])).convert("RGB"), best[2]


def ocr_words(
    image: Image.Image, *, psm: int = 6, scale: float = 1.0, binarise: bool = False
) -> list[Word]:
    """OCR one image, returning word boxes in the original pixel frame."""
    if scale != 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    if binarise:
        grey = np.asarray(image.convert("L"))
        image = Image.fromarray(np.where(grey < 170, 0, 255).astype(np.uint8))
    out = _tesseract(image, psm=psm, whitelist=None)
    words: list[Word] = []
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12 or not parts[11].strip():
            continue
        try:
            conf = float(parts[10])
        except ValueError:
            continue
        if conf < 0:
            continue
        words.append(
            Word(
                left=round(int(parts[6]) / scale),
                top=round(int(parts[7]) / scale),
                width=round(int(parts[8]) / scale),
                height=round(int(parts[9]) / scale),
                text=parts[11].strip(),
                conf=conf,
            )
        )
    return words


def tesseract_languages() -> set[str]:
    """The language packs the installed tesseract can load, by name.

    Empty when tesseract is not on PATH at all. Callers use this to tell a
    missing OCR prerequisite (skip, or refuse with the install command) from a
    timetable that the OCR genuinely failed to read.
    """
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip() and ":" not in line}


def _tesseract(image: Image.Image, *, psm: int, whitelist: str | None) -> str:
    command = ["tesseract", "stdin", "stdout", "--psm", str(psm), "-l", OCR_LANGUAGE, "tsv"]
    if whitelist:
        command += ["-c", f"tessedit_char_whitelist={whitelist}"]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    try:
        result = subprocess.run(command, input=buffer.getvalue(), capture_output=True, check=True)
    except OSError as error:
        raise RuntimeError(f"tesseract is not installed or not on PATH: {error}") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(_ocr_error(error.stderr, "tesseract failed")) from error
    output = result.stdout.decode("utf-8", errors="replace")
    if not output.startswith(TSV_HEADER):
        # tesseract exits 0 without TSV when it cannot read a file it was asked
        # for: a `tsv` config missing from TESSDATA_PREFIX makes it fall back to
        # plain text, which parses into no words at all. That is otherwise
        # indistinguishable from a cell that reads nothing, and it degrades a
        # whole build silently — every caption comes back empty, so no band is
        # found and each PDF is skipped as unreadable. A blank crop still returns
        # the header, so its absence is always a failure.
        raise RuntimeError(_ocr_error(result.stderr, "tesseract produced no TSV"))
    return output


def _ocr_error(stderr: bytes, lead: str) -> str:
    """The message for a tesseract run that did not produce a reading."""
    detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
    reason = f": {detail[-1]}" if detail else ""
    return (
        f"{lead}{reason}. It needs both the '{OCR_LANGUAGE}' language pack and the "
        f"tsv config (Debian/Ubuntu: {OCR_LANGUAGE_PACKAGE})"
    )


def read_cell(image: Image.Image, box: tuple[int, int, int, int]) -> str:
    """Read a single time out of one cell, with a digits-and-colon whitelist."""
    x0, y0, x1, y1 = box
    crop = image.crop((max(0, x0), max(0, y0), x1, y1)).convert("L")
    if crop.width < 4 or crop.height < 4:
        return ""
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
    crop = Image.fromarray(np.where(np.asarray(crop) < 150, 0, 255).astype(np.uint8))
    out = _tesseract(crop, psm=7, whitelist=TIME_WHITELIST)
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 12 and parts[11].strip():
            text = parts[11].strip()
            if TIME.match(text) or DASH.match(text):
                return text
    return ""


def _polarities(crop: Image.Image) -> list[Image.Image]:
    """Black-on-white renderings of a caption crop, one per polarity.

    A header band is a light colour on some lines and a dark one on others, so
    inverting blindly turns white-on-blue into black-on-dark. Both polarities
    are produced and the caller keeps whichever OCRs into a recognisable word.
    """
    grey = np.asarray(crop.convert("L")).astype(np.int16)
    low, high = int(grey.min()), int(grey.max())
    if high - low < 30:
        return [crop.convert("L")]
    middle = (low + high) / 2
    return [
        Image.fromarray(np.where(grey < middle, 0, 255).astype(np.uint8)),
        Image.fromarray(np.where(grey > middle, 0, 255).astype(np.uint8)),
    ]


def read_caption(
    image: Image.Image, box: tuple[int, int, int, int]
) -> tuple[str, list[tuple[int, int, str]]]:
    """OCR one caption crop, returning the text and its word boxes."""
    best: tuple[float, str, list[tuple[int, int, str]]] = (-1.0, "", [])
    for candidate in _polarities(image.crop(box)):
        out = _tesseract(candidate, psm=7, whitelist=None)
        words = [
            (int(parts[6]), int(parts[6]) + int(parts[8]), parts[11].strip())
            for parts in (line.split("\t") for line in out.splitlines()[1:])
            if len(parts) >= 12 and parts[11].strip() and WORDISH.search(parts[11])
        ]
        text = " ".join(word[2] for word in words).strip()
        score = 2.0 if SERVICE_HEADER.search(unaccent(text)) else 0.0
        score += len(text) / 1000
        if score > best[0]:
            best = (score, text, words)
    return best[1], best[2]


def banner_text(pdf_path: Path, rect: pymupdf.Rect) -> str:
    """OCR the vector route caption printed above the timetable bitmap."""
    if rect.y0 < 12:
        return ""
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        clip = pymupdf.Rect(0, max(0.0, rect.y0 - 45), page.rect.x1, rect.y0 - 3)
        pixmap = page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(6, 6), colorspace=pymupdf.csGRAY)
        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    out = _tesseract(image, psm=6, whitelist=None)
    words = [
        parts[11] for parts in (row.split("\t") for row in out.splitlines()[1:]) if len(parts) >= 12
    ]
    return " ".join(words).strip()


# -- pixel layout ----------------------------------------------------------


def close_horizontal(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fill horizontal holes up to ``radius`` wide, to bridge glyph gaps."""
    sums = np.pad(np.cumsum(mask, axis=1), ((0, 0), (1, 0)))
    width = mask.shape[1]
    xs = np.arange(width)
    return (
        sums[:, np.clip(xs + radius + 1, 0, width)] - sums[:, np.clip(xs - radius, 0, width)]
    ) > 0


def runs(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    """Maximal runs of True at least ``minimum`` long."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum:
                spans.append((start, index))
            start = None
    if start is not None and len(mask) - start >= minimum:
        spans.append((start, len(mask)))
    return spans


def close_vertical(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fill vertical holes up to ``radius`` tall, to bridge glyph gaps."""
    sums = np.pad(np.cumsum(mask, axis=0), ((1, 0), (0, 0)))
    height = mask.shape[0]
    ys = np.arange(height)
    return (sums[np.clip(ys + radius + 1, 0, height)] - sums[np.clip(ys - radius, 0, height)]) > 0


def header_bands(image: Image.Image) -> list[Band]:
    """The coloured column-group headers, whatever colour the line uses.

    A row counts as band when a fifth of it is coloured; consecutive such rows,
    allowing four rows of white caption between them, form one band. The row
    profile is taken from the raw mask rather than a closed one, so the band
    keeps its true height instead of growing into the timetable beneath it.
    """
    rgb = np.asarray(image).astype(np.int16)
    coloured = (rgb.max(axis=2) - rgb.min(axis=2)) > SATURATION
    rows = np.where(coloured.mean(axis=1) > BAND_FRACTION)[0]
    bands: list[Band] = []
    for group in _contiguous(rows, gap=4):
        top, bottom = int(group[0]), int(group[-1]) + 1
        if not 10 <= bottom - top <= 200:
            continue
        columns = np.where(coloured[top:bottom].mean(axis=0) > 0.5)[0]
        if len(columns) < 40:
            continue
        bands.append(Band(top, bottom, int(columns[0]), int(columns[-1]) + 1))
    return bands


def _contiguous(values: np.ndarray, gap: int) -> list[list[int]]:
    """Split sorted integers wherever the step exceeds ``gap``."""
    groups: list[list[int]] = []
    for value in values:
        value = int(value)
        if groups and value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def keep_heading_bands(bands: list[Band], words: list[Word], bottom: int) -> list[Band]:
    """Drop colour bars that head no timetable.

    Every page ends with coloured footnote swatches that look exactly like a
    header band. A real one has at least three timed rows underneath it before
    the next band starts.
    """
    kept: list[Band] = []
    for band in bands:
        below = [other.y0 for other in bands if other.y0 > band.y0]
        end = min(below) if below else bottom
        timed = [word for word in words if band.y1 < word.y < end and TIME.match(word.text)]
        if len(timed) >= 3:
            kept.append(band)
    return kept


def column_highlights(
    image: Image.Image, top: int, bottom: int, columns: list[float], pitch: float
) -> tuple[list[str], set[int]]:
    """The background colour of each trip column, and which ones are tinted.

    Operators mark a restricted service by tinting the column instead of
    writing it down: lavender on line 1, salmon, yellow and blue on line 12.
    The tint is therefore read off the pixels. A tinted column is one whose
    background is saturated, which the grey and white row shading never is.
    """
    rgb = np.asarray(image).astype(np.int16)
    colours: list[str] = []
    highlighted: set[int] = set()
    for index, centre in enumerate(columns):
        x0 = max(0, int(centre - pitch * 0.18))
        x1 = min(rgb.shape[1], int(centre + pitch * 0.18))
        patch = rgb[int(top) : int(bottom), x0:x1].reshape(-1, 3)
        background = patch[patch.max(axis=1) > 90] if len(patch) else patch
        if not len(background):
            colours.append("")
            continue
        quantised = (background // 8) * 8
        values, counts = np.unique(quantised, axis=0, return_counts=True)
        mode = values[counts.argmax()]
        colours.append("#{:02x}{:02x}{:02x}".format(*(int(v) for v in mode)))
        saturated = (background.max(axis=1) - background.min(axis=1)) > 30
        if float(saturated.mean()) > 0.5:
            highlighted.add(index)
    return colours, highlighted


# -- grid reconstruction ---------------------------------------------------


def cluster(values: list[float], tolerance: float) -> list[tuple[float, int]]:
    """Group values into ``(centre, size)``, splitting where the gap is too big."""
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(sum(g) / len(g), len(g)) for g in groups]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def column_pitch(time_words: list[Word], row_tolerance: float) -> float:
    """The spacing between adjacent trip columns, from the closest pair in any row."""
    rows: dict[int, list[float]] = {}
    for word in time_words:
        rows.setdefault(round(word.y / row_tolerance), []).append(word.x)
    deltas: list[float] = []
    for xs in rows.values():
        ordered = sorted(xs)
        gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 8]
        if gaps:
            deltas.append(min(gaps))
    if deltas:
        return median(deltas)
    widths = [w.width for w in time_words]
    return median(widths) * 1.25 if widths else 60.0


def build_block(
    image: Image.Image | None,
    words: list[Word],
    bands: list[Band],
    index: int,
    top: int,
    bottom: int,
) -> Block | None:
    """Recover the stop list, the trip columns and every time of one block."""
    inside = [
        word
        for word in words
        if top < word.y < bottom and not any(band.y0 - 4 <= word.y <= band.y1 + 4 for band in bands)
    ]
    time_words = [word for word in inside if TIME.match(word.text)]
    if not time_words:
        return None

    height = median([float(word.height) for word in inside])
    row_tolerance = max(6.0, height * 0.80)
    pitch = column_pitch(time_words, row_tolerance * 2)

    columns = [centre for centre, _ in cluster([word.x for word in time_words], pitch * 0.45)]
    if not columns:
        return None
    first_column = min(columns) - pitch * 0.5

    # Rows come from the times and dashes, but also from the stop names: a stop
    # whose every time the OCR failed on is still a stop of the line, and
    # leaving it out would silently drop it from the OD pairs.
    name_words = [
        word
        for word in inside
        if word.x < first_column and WORDISH.search(word.text) and not TIME.match(word.text)
    ]
    last_timed = max(word.y for word in time_words)
    rows = [
        centre
        for centre, _ in cluster(
            [word.y for word in time_words + [w for w in inside if DASH.match(w.text)]],
            row_tolerance,
        )
        if centre <= last_timed + row_tolerance
    ]
    first_row = min(rows) if rows else min((word.y for word in name_words), default=0)
    for centre, _ in cluster([word.y for word in name_words], row_tolerance):
        if centre > last_timed + row_tolerance:
            continue
        if centre < first_row - row_tolerance * 2:
            continue
        if all(abs(centre - existing) > row_tolerance * 0.5 for existing in rows):
            rows.append(centre)
    rows.sort()
    if not rows:
        return None

    matrix = [["" for _ in columns] for _ in rows]
    for word in time_words:
        row = min(range(len(rows)), key=lambda i: abs(rows[i] - word.y))
        column = min(range(len(columns)), key=lambda j: abs(columns[j] - word.x))
        if abs(rows[row] - word.y) > row_tolerance or abs(columns[column] - word.x) > pitch * 0.5:
            continue
        matrix[row][column] = word.text

    stops: list[str] = []
    for centre in rows:
        names = sorted(
            (
                word
                for word in inside
                if abs(word.y - centre) <= row_tolerance
                and word.x < first_column
                and WORDISH.search(word.text)
            ),
            key=lambda word: word.left,
        )
        if len(names) > 1 and len(names[0].text) == 1 and names[0].right + 8 < names[1].left:
            names = names[1:]  # the row's red pictogram, read as a stray letter
        stops.append(" ".join(word.text for word in names).strip())

    if image is not None and rows:
        colours, highlight = column_highlights(image, rows[0], rows[-1] + 1, columns, pitch)
    else:
        colours, highlight = [""] * len(columns), set()
    return Block(
        index=index,
        top=top,
        bottom=bottom,
        columns=columns,
        rows=rows,
        stops=stops,
        times=matrix,
        bands=list(bands),
        highlighted=highlight,
        colours=colours,
        repaired=0,
    )


# -- reconciliation --------------------------------------------------------


def cell_box(
    block: Block, row: int, column: int, pitch: float, pad: int = 6
) -> tuple[int, int, int, int]:
    """The pixel rectangle of one (row, column) cell, with a little padding."""

    def bounds(centres: list[float], index: int, half: float, lo: int, hi: int) -> tuple[int, int]:
        low = centres[index] - half if index == 0 else (centres[index - 1] + centres[index]) / 2
        high = (
            centres[index] + half
            if index == len(centres) - 1
            else (centres[index] + centres[index + 1]) / 2
        )
        return max(lo, round(low) - pad), min(hi, round(high) + pad)

    x0, x1 = bounds(block.columns, column, pitch * 0.5, 0, 10**9)
    half = (block.rows[1] - block.rows[0]) / 2 if len(block.rows) > 1 else 20
    y0, y1 = bounds(block.rows, row, half, block.top, block.bottom)
    return x0, y0, x1, y1


def assign_matrix(block: Block, words: list[Word]) -> list[list[str]]:
    """Map an independent OCR pass onto the grid the first pass established."""
    tolerance = (block.rows[1] - block.rows[0]) / 2 if len(block.rows) > 1 else 20
    pitch = block.columns[1] - block.columns[0] if len(block.columns) > 1 else 60
    matrix = [["" for _ in block.columns] for _ in block.rows]
    for word in words:
        if not TIME.match(word.text):
            continue
        row = min(range(len(block.rows)), key=lambda i: abs(block.rows[i] - word.y))
        column = min(range(len(block.columns)), key=lambda j: abs(block.columns[j] - word.x))
        if (
            abs(block.rows[row] - word.y) <= tolerance
            and abs(block.columns[column] - word.x) <= pitch * 0.5
        ):
            matrix[row][column] = word.text
    return matrix


def reconcile(image: Image.Image, block: Block, second: list[list[str]]) -> None:
    """Settle each cell between two OCR passes, asking a third when they differ.

    A cell is never emptied: an empty reading is much more likely than a wrong
    one to be the OCR's fault, so the crop is asked, then the first pass, then
    the second, and a cell the passes could not agree on is recorded instead of
    discarded.
    """
    pitch = block.columns[1] - block.columns[0] if len(block.columns) > 1 else 60
    for row in range(len(block.rows)):
        for column in range(len(block.columns)):
            first = block.times[row][column]
            other = second[row][column]
            if first == other:
                continue
            third = read_cell(image, cell_box(block, row, column, pitch))
            chosen = third or first or other
            if not third and first and other:
                block.warnings.append(
                    f"cell r{row} c{column}: passes disagreed "
                    f"({first!r} / {other!r}), kept {chosen!r}"
                )
            if chosen != first:
                block.repaired += 1
            block.times[row][column] = chosen


def fill_missing(image: Image.Image, block: Block) -> None:
    """Read every cell the page-level OCR left empty, one crop at a time.

    An empty cell is ambiguous — the trip may not call there, or the page-level
    OCR may simply have missed the time. Asking each empty cell directly turns
    the guess into a reading: a crop returns either a time, a printed dash, or
    nothing at all, and only the first of those is a call.
    """
    pitch = block.columns[1] - block.columns[0] if len(block.columns) > 1 else 60
    for row in range(len(block.rows)):
        for column in range(len(block.columns)):
            if block.times[row][column]:
                continue
            text = read_cell(image, cell_box(block, row, column, pitch))
            if text and TIME.match(text):
                block.times[row][column] = text
                block.repaired += 1


def enforce_monotonic(
    image: Image.Image | None, block: Block, alternatives: list[list[str]] | None = None
) -> None:
    """A trip's times never run backwards; repair or flag any cell that does.

    A breach means the cell was misread, so the cell is re-read from a crop of
    itself and the independent second pass is consulted; the first candidate
    that puts the column back in order wins. Only when none does is the
    original kept and the breach recorded, because deleting a time loses more
    than it corrects.
    """
    pitch = block.columns[1] - block.columns[0] if len(block.columns) > 1 else 60
    for column in range(len(block.columns)):
        previous: int | None = None
        for row in range(len(block.rows)):
            current = block.times[row][column]
            minutes = to_minutes(current)
            if minutes is None:
                continue
            if previous is not None and minutes < previous:
                candidates = [
                    read_cell(image, cell_box(block, row, column, pitch)) if image else ""
                ]
                if alternatives:
                    candidates.append(alternatives[row][column])
                follow = (
                    to_minutes(block.times[row + 1][column]) if row + 1 < len(block.rows) else None
                )
                for candidate in candidates:
                    value = to_minutes(candidate) if candidate else None
                    if value is None or value < previous or (follow is not None and value > follow):
                        continue
                    block.times[row][column] = candidate
                    block.repaired += 1
                    minutes = value
                    break
                else:
                    block.warnings.append(
                        f"r{row} c{column}: {current} follows {format_minutes(previous)} "
                        f"at {block.stops[row] or '?'}"
                    )
            previous = minutes


def format_minutes(minutes: int) -> str:
    """Minutes past midnight as ``HH:MM``, for diagnostics."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def roll_block(block: Block) -> int:
    """Put a block's columns on a continuous scale across midnight.

    The last trips of the day are printed as ``23:58`` then ``00:02``, which
    read literally runs backwards. The grid is corrected before the ordering
    invariant is checked, so a genuine breach is not lost in a false one.
    """
    crossings = 0
    for column in range(len(block.columns)):
        offset = 0
        previous: int | None = None
        for row in range(len(block.rows)):
            minutes = to_minutes(block.times[row][column])
            if minutes is None:
                continue
            minutes += offset
            if previous is not None and minutes < previous - 12 * 60:
                offset += 24 * 60
                minutes += 24 * 60
                crossings += 1
            block.times[row][column] = format_minutes(minutes)
            previous = minutes
    return crossings


def to_minutes(text: str) -> int | None:
    """``"07:45"`` to minutes past midnight; ``None`` when there is no time."""
    match = TIME.match(text.strip()) if text else None
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


# -- extraction ------------------------------------------------------------


@dataclass
class Extraction:
    """One parsed PDF, with how much work the OCR needed."""

    line: str
    source: str
    caption: str
    trips: list[Trip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repaired: int = 0
    crossings: int = 0


def label_bands(image: Image.Image, bands: list[Band]) -> list[Band]:
    """Split each coloured strip into one band per printed caption.

    Some lines print both service captions inside a single strip
    (``DIAS ÚTEIS ... FIM DE SEMANA E FERIADOS``), so a strip is not one group.
    Each strip's caption words are read back with their boxes and cut into runs
    wherever the gap between two words is far wider than the gap between words.

    The caption is white on colour, which the default tesseract binarisation
    reads as background, so the strip is cropped and inverted before OCR.
    """
    split: list[Band] = []
    for band in bands:
        _, words = read_caption(image, (band.x0, band.y0, band.x1, band.y1 + 1))
        if not words:
            continue
        gaps = [b[0] - a[1] for a, b in zip(words, words[1:]) if b[0] - a[1] > 2]
        threshold = max(4 * median(gaps), 40) if gaps else 10**9
        runs: list[list[tuple[int, int, str]]] = []
        for word in words:
            if runs and word[0] - runs[-1][-1][1] <= threshold:
                runs[-1].append(word)
            else:
                runs.append([word])
        # A stray digit inside the strip is not a caption.
        runs = [run for run in runs if run[-1][1] - run[0][0] >= 40]
        # Captions partition the strip: each one governs the columns around it,
        # not just the few pixels its own text covers.
        centres = [(run[0][0] + run[-1][1]) / 2 for run in runs]
        for index, run in enumerate(runs):
            left = (
                band.x0
                if index == 0
                else band.x0 + round((centres[index - 1] + centres[index]) / 2)
            )
            right = (
                band.x1
                if index + 1 == len(runs)
                else band.x0 + round((centres[index] + centres[index + 1]) / 2)
            )
            split.append(
                Band(
                    y0=band.y0,
                    y1=band.y1,
                    x0=left,
                    x1=right,
                    label=" ".join(word[2] for word in run),
                )
            )
    return split


def text_words(page: pymupdf.Page) -> list[Word]:
    """Exact word boxes from a PDF text layer, in page points.

    Two of the fourteen line PDFs carry a real text layer instead of a bitmap,
    so the same grid reconstruction is fed from the text layer there.
    """
    return [
        Word(
            left=round(x0),
            top=round(y0),
            width=round(x1 - x0),
            height=round(y1 - y0),
            text=word,
            conf=100.0,
        )
        for x0, y0, x1, y1, word, *_ in page.get_text("words")
        if word.strip()
    ]


def text_bands(words: list[Word]) -> list[Band]:
    """Service headers of a text-layer timetable, one band per header line."""
    bands: list[Band] = []
    for centre, _ in cluster([word.y for word in words], 3.0):
        line = sorted((word for word in words if abs(word.y - centre) <= 3.5), key=lambda w: w.left)
        caption = " ".join(word.text for word in line)
        if not line or not SERVICE_HEADER.search(unaccent(caption)):
            continue
        bands.append(
            Band(
                y0=min(word.top for word in line),
                y1=max(word.bottom for word in line),
                x0=min(word.left for word in line),
                x1=max(word.right for word in line),
                label=caption,
            )
        )
    return bands


def text_caption(words: list[Word], band: Band) -> str:
    """The route description of a text-layer timetable.

    The line naming the places the service calls at is the first one above the
    timetables that separates them, as in ``AVEIRO | FORTE DA BARRA | S. JACINTO``.
    """
    lines: dict[float, list[Word]] = {}
    for word in words:
        lines.setdefault(round(word.y / 3) * 3, []).append(word)
    above = [
        sorted(group, key=lambda w: w.left)
        for centre, group in sorted(lines.items())
        if centre < band.y0
    ]
    for line in above:
        text = " ".join(word.text for word in line)
        if text.count("|") >= 2:
            return text
    return " ".join(word.text for word in above[-1]) if above else ""


def parse_text_pdf(pdf_path: Path, *, operator: str, line: str) -> Extraction:
    """Every trip of a line PDF whose timetable is a real text layer."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        words = text_words(page)
        height = page.rect.height
    bands = text_bands(words)
    if not bands:
        raise ValueError(f"no service headers found in {pdf_path}")
    caption = text_caption(words, bands[0])
    extraction = Extraction(line=line or pdf_path.stem, source=pdf_path.name, caption=caption)
    for index, band in enumerate(bands):
        bottom = bands[index + 1].y0 if index + 1 < len(bands) else round(height)
        block = build_block(None, words, [band], index, band.y1, bottom)
        if block is None:
            extraction.warnings.append(f"block {index + 1}: no grid found")
            continue
        extraction.crossings += roll_block(block)
        enforce_monotonic(None, block)
        extraction.repaired += block.repaired
        extraction.warnings.extend(f"block {index + 1}: {w}" for w in block.warnings)
        extraction.trips.extend(_trips(block, extracting=extraction))
    return extraction


def parse_pdf(pdf_path: Path, *, operator: str = "aveirobus", line: str = "") -> Extraction:
    """Every trip of one AveiroBus line PDF.

    Bitmap timetables are OCR'd in two passes; text-layer timetables need no
    OCR at all and their words are exact.
    """
    try:
        image, rect = timetable_image(pdf_path)
    except (ValueError, KeyError):
        return parse_text_pdf(pdf_path, operator=operator, line=line)
    first = ocr_words(image, psm=6)
    second = ocr_words(image, psm=4, scale=1.6, binarise=True)

    strips = label_bands(image, header_bands(image))
    # A strip is only a header if it carries a service caption. On line 12 the
    # alternating row shading is itself a saturated colour, so height and width
    # alone would mistake every shaded row for a header.
    captioned = [band for band in strips if SERVICE_HEADER.search(unaccent(band.label))]
    bands = keep_heading_bands(captioned or strips, first, image.height)
    if not bands:
        raise ValueError(f"no header bands found in {pdf_path}")
    headers: list[list[Band]] = []
    for band in sorted(bands, key=lambda b: (b.y0, b.x0)):
        for group in headers:
            if abs(group[0].y0 - band.y0) <= 18:
                group.append(band)
                break
        else:
            headers.append([band])

    extraction = Extraction(
        line=line or pdf_path.stem, source=pdf_path.name, caption=banner_text(pdf_path, rect)
    )

    for index, group in enumerate(headers):
        top = max(band.y1 for band in group)
        bottom = headers[index + 1][0].y0 if index + 1 < len(headers) else image.height
        block = build_block(image, first, group, index, top, bottom)
        if block is None:
            extraction.warnings.append(f"block {index + 1}: no grid found")
            continue
        alternatives = assign_matrix(block, second)
        reconcile(image, block, alternatives)
        fill_missing(image, block)
        extraction.crossings += roll_block(block)
        enforce_monotonic(image, block, alternatives)
        extraction.repaired += block.repaired
        extraction.warnings.extend(f"block {index + 1}: {w}" for w in block.warnings)
        extraction.trips.extend(_trips(block, extracting=extraction))
    return extraction


def _trips(block: Block, *, extracting: Extraction) -> list[Trip]:
    """Turn one recovered grid into one trip per column."""
    trips: list[Trip] = []
    for column in range(len(block.columns)):
        service = block.service_of(column)
        calls = [
            Call(
                stop_seq=row + 1,
                stop=block.stops[row],
                time=block.times[row][column],
                minutes=to_minutes(block.times[row][column]),
                markers="!*" if column in block.highlighted else "",
            )
            for row in range(len(block.rows))
        ]
        if not any(call.minutes is not None for call in calls):
            continue
        direction = f"b{block.index + 1}"
        trips.append(
            Trip(
                trip_id=f"{'aveirobus'}:{extracting.line}:{direction}:{service}:{column + 1}",
                operator="aveirobus",
                line=extracting.line,
                direction=direction,
                service=service,
                season="",
                trip_index=column + 1,
                source=extracting.source,
                notes=extracting.caption,
                block=block.index + 1,
                school_only=column in block.highlighted,
                colour=block.colours[column] if column < len(block.colours) else "",
                calls=calls,
            )
        )
    return trips
