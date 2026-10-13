"""Parse BusWay line pages into trips and their stop-to-stop calls.

Every BusWay line page carries one ``<table class="tabela-horaria">`` per
service type (weekdays, Saturdays, Sundays and holidays). Inside such a table
**each column is one trip** and each row is one stop, so a cell is a single
departure time — or ``–`` when that trip does not call at that stop. The page
therefore publishes the whole timetable, not just endpoint departures, which is
what an OD matrix needs.

Cell attributes carry the service detail:

``ÉPOCA``
    ``A`` = *Anual*, ``PE`` = *Período Escolar* (term time only).
``fundoamarelo``
    The cell is highlighted because it is the hand-over point of a connection;
    the matching footnote in the table's ``observacao`` row says to which line.
``*``
    A marker pointing at that same footnote.

The parser is deliberately ignorant of *which* lines exist; it turns one page
into records and leaves the inventory to `inmotion.bus.sources`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from inmotion.bus import records
from inmotion.bus.records import Call, Trip

SERVICES = {
    # Checked first: "Fins de Semana e Feriados" is the non-weekday service, and
    # matching "feriados" alone would file it as holidays only.
    "fins de semana": "weekend",
    "dias úteis": "weekday",
    "sábados": "saturday",
    "domingos e feriados": "sunday",
    "domingos": "sunday",
    "feriados": "holiday",
}

_TIME = re.compile(r"(\d{1,2})\s*[:.]\s*(\d{2})")
_NOTE_MARK = re.compile(r"[*†‡]+")


@dataclass
class Cell:
    """One table cell: its text, its CSS classes and its column span."""

    text: str = ""
    classes: tuple[str, ...] = ()
    colspan: int = 1

    def has(self, name: str) -> bool:
        """Whether the cell carries exactly this class."""
        return name in self.classes

    def starts(self, prefix: str) -> bool:
        """Whether any class starts with this prefix.

        BusWay ships two table templates: one names its classes
        ``epoca_1_20``, the other ``epoca_M_Sever_Vouga``. Matching the prefix
        covers both without enumerating the variants.
        """
        return any(name.startswith(prefix) for name in self.classes)


class _Tables(HTMLParser):
    """Collect ``<table>`` elements as nested lists of cells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[tuple[str, list[list[Cell]]]] = []
        self._stack: list[list[list[Cell]]] = []
        self._row: list[Cell] | None = None
        self._cell: Cell | None = None
        self._buf: list[str] = []
        self._classes: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "table":
            self._stack.append([])
            self._classes.append(attr.get("class") or "")
        elif tag == "tr" and self._stack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = Cell(
                classes=tuple((attr.get("class") or "").split()),
                colspan=int(attr.get("colspan") or 1),
            )
            self._buf = []
        elif tag == "br" and self._cell is not None:
            self._buf.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            self._cell.text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None and self._stack:
            self._stack[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._stack:
            rows = self._stack.pop()
            classes = self._classes.pop()
            if self._stack:
                self._stack[-1].append([])  # keep nesting harmless
            self.tables.append((classes, rows))


def parse_tables(html: str) -> list[tuple[str, list[list[Cell]]]]:
    """Every table on the page, with its class attribute."""
    parser = _Tables()
    parser.feed(html)
    return parser.tables


def _service(text: str) -> str:
    low = text.lower()
    for prefix, name in SERVICES.items():
        if prefix in low:
            return name
    return "unknown"


def _time(text: str) -> tuple[str, int | None]:
    """Split a cell into ``HH:MM`` and minutes past midnight, if it has one."""
    match = _TIME.search(text)
    if not match:
        return "", None
    hours, minutes = int(match.group(1)), int(match.group(2))
    return f"{hours:02d}:{minutes:02d}", hours * 60 + minutes


def page_identity(html: str, *, line: str, direction: str) -> tuple[str, str, str]:
    """Read the line, direction and route caption a page carries.

    The URL slug wins over the heading: the summer line is published as both
    ``l35`` and ``35-volta``, and the ``15-volta`` page's heading says ``IDA``
    because it was copy-pasted. The heading is only the caption, plus a
    fallback for a slug that names no direction at all.
    """
    for classes, rows in parse_tables(html):
        for row in rows:
            for cell in row:
                if cell.starts("numero_linha"):
                    parts = cell.text.replace("\u2013", "-").split("-", 1)
                    rest = parts[1].strip().split() if len(parts) > 1 else []
                    found = rest[0].lower() if rest else ""
                    return (
                        line or parts[0].strip().lstrip("lL"),
                        direction or (found if found in ("ida", "volta") else direction),
                        cell.text.strip(),
                    )
    return line, direction, ""


def parse_line_page(
    html: str,
    *,
    operator: str,
    line: str,
    direction: str,
    source: str,
    page_key: str,
) -> list[Trip]:
    """Every trip on one BusWay line page, for the direction it carries.

    ``page_key`` is the page's own slug and is what makes a trip id unique: two
    pages of the same line can claim the same direction, and then only the slug
    tells their trips apart.
    """
    line, direction, _caption = page_identity(html, line=line, direction=direction)
    trips: list[Trip] = []
    for classes, rows in parse_tables(html):
        if "tabela-horaria" not in classes.split():
            continue
        seasons: list[str] = []
        service = "unknown"
        notes = ""
        stop_rows: list[list[Cell]] = []
        for row in rows:
            if any(cell.has("funcionamento") for cell in row):
                service = _service(" ".join(c.text for c in row if c.has("funcionamento")))
            header = [c.text for c in row if c.starts("epoca") and not c.starts("epoca_titulo")]
            if header:
                seasons = header
            if any(c.has("paragem") for c in row):
                stop_rows.append(row)
            for cell in row:
                if "observac" in " ".join(cell.classes):
                    notes = f"{notes} {cell.text}".strip()
        if not seasons or not stop_rows:
            continue

        width = len(seasons)
        # One list per table: a page carries a table per service type, and each
        # names its own trips from column 1, so indexing the page-wide list
        # would append one service's calls onto another's trips.
        table_trips = [
            Trip(
                trip_id=f"{operator}:{page_key}:{service}:{index + 1}",
                operator=operator,
                line=line,
                direction=direction,
                service=service,
                season=season,
                trip_index=index + 1,
                source=source,
                notes=notes,
            )
            for index, season in enumerate(seasons)
        ]
        trips.extend(table_trips)

        for seq, row in enumerate(stop_rows, start=1):
            stop = next(c.text for c in row if c.has("paragem"))
            times = [c for c in row if not c.has("paragem") and not c.has("seta")]
            if len(times) != width:
                continue
            for index, cell in enumerate(times):
                time, minutes = _time(cell.text)
                markers = _NOTE_MARK.findall(cell.text)
                if cell.has("fundoamarelo"):
                    markers.append("!")
                table_trips[index].calls.append(
                    Call(
                        stop_seq=seq,
                        stop=stop,
                        time=time,
                        minutes=minutes,
                        markers="".join(markers),
                    )
                )
    for trip in trips:
        records.roll_past_midnight(trip)
    return trips
