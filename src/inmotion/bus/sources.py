"""Polite, cached, checksummed fetching of the operator sites, plus inventory.

Neither operator publishes GTFS, so the schedules only exist as web pages. This
module is the one place that talks to the two sites:

* it sleeps ``crawl_delay`` seconds between network requests, using the value
  each site's ``robots.txt`` asks for (AveiroBus asks for 10s, BusWay for none);
* it caches every response under ``data/bus/raw/<operator>/`` and serves repeat
  runs from disk, so a re-parse never re-hits the site;
* it records url, sha256 and size of every artifact in ``manifest.json``, which
  makes the scrape replayable and detects a page that changed under our feet.

Nothing here parses timetables; that is `inmotion.bus.busway` and
`inmotion.bus.aveirobus`.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from inmotion.bus import paths

USER_AGENT = (
    "inMotion-research/0.1 (academic study of urban mobility; "
    "https://github.com/inmotion; contact andrepedroribeiro@av.it.pt)"
)

RETRIES = 3


@dataclass(frozen=True)
class Operator:
    """One bus operator site."""

    key: str
    name: str
    base: str
    crawl_delay: float


BUSWAY = Operator("busway", "BusWay (CIRA)", "https://busway-cira.pt", 1.0)
AVEIROBUS = Operator("aveirobus", "AveiroBus", "https://www.aveirobus.pt", 10.0)
OPERATORS = (BUSWAY, AVEIROBUS)

# BusWay's own sitemap is the authoritative list of line pages. Crawling the
# per-municipality pages instead reaches barely half of them: those pages link
# one direction only, slug the summer line two different ways (``l35`` and
# ``35-volta``), and omit some pages entirely.
BUSWAY_SITEMAP = f"{BUSWAY.base}/page-sitemap.xml"

# AveiroBus publishes one timetable PDF per urban line; this is its index page.
AVEIROBUS_INDEX = f"{AVEIROBUS.base}/circuito-urbano"
AVEIROBUS_EXTRA_INDEXES = (f"{AVEIROBUS.base}/longo-curso",)


@dataclass(frozen=True)
class LinePage:
    """One served timetable page: a line, in one direction."""

    operator: str
    line: str
    direction: str
    url: str
    path: Path


class Fetcher:
    """Fetch with a cache, a crawl delay, and a manifest of what was fetched."""

    def __init__(self, *, force: bool = False, verbose: bool = True) -> None:
        self.force = force
        self.verbose = verbose
        self._entry: dict[str, dict[str, object]] = {}
        self._last_request = 0.0
        if paths.MANIFEST.exists():
            self._entry = json.loads(paths.MANIFEST.read_text())

    # -- transport ---------------------------------------------------------

    def _sleep(self, operator: Operator) -> None:
        wait = operator.crawl_delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _download(self, url: str, target: Path, operator: Operator) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last: Exception | None = None
        for attempt in range(RETRIES):
            self._sleep(operator)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                self._last_request = time.monotonic()
                return body
            except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover
                last = exc
                self._last_request = time.monotonic()
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {url}: {last}")

    def get(self, url: str, operator: Operator, *, suffix: str = ".html") -> Path:
        """Return the local path of ``url``, downloading it once."""
        name = local_name(url, suffix=suffix)
        target = paths.raw_dir(operator.key) / name

        known = self._entry.get(url)
        if target.exists() and known and not self.force:
            size = target.stat().st_size
            if int(known["bytes"]) == size:  # type: ignore[call-overload]
                if self.verbose:
                    print(f"  cached  {name}")
                return target

        body = self._download(url, target, operator)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        previous = known.get("sha256") if known else None
        if previous and previous != digest and self.verbose:
            print(f"  CHANGED {name} ({previous[:8]} -> {digest[:8]})")
        self._entry[url] = {
            "operator": operator.key,
            "path": str(target.relative_to(paths.REPO_ROOT)),
            "bytes": len(body),
            "sha256": digest,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if self.verbose:
            print(f"  fetched {name} ({len(body) / 1024:.0f} KiB)")
        return target

    def close(self) -> None:
        """Persist the manifest, oldest entries first."""
        paths.RAW_DIR.mkdir(parents=True, exist_ok=True)
        ordered = dict(sorted(self._entry.items(), key=lambda kv: str(kv[1]["fetched_at"])))
        paths.MANIFEST.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# -- inventory -------------------------------------------------------------

_SITEMAP_LOC = re.compile(r"<loc>\s*(https://busway-cira\.pt/linhas/([^/<\s]+)/)\s*</loc>")
_AVEIROBUS_PDF = re.compile(
    r'<a[^>]+href="([^"]*?/sites/default/files/[^"]*?\.pdf)"[^>]*>(.*?)</a>', re.S
)
_TAGS = re.compile(r"<[^>]+>")


def local_name(url: str, *, suffix: str = ".html") -> str:
    """A readable, collision-free filename for a URL.

    ``/linhas/16-ida/`` becomes ``16-ida.html`` and a PDF keeps its own name, so
    the raw directory can be read by a human debugging a parse.
    """
    path = urllib.parse.urlsplit(url).path
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return f"index{suffix}"
    last = segments[-1]
    name = last if "." in last else "-".join(segments[-2:])
    if "." not in name:
        name += suffix
    return name


def parse_line_id(line_id: str) -> tuple[str, str]:
    """Split ``16-ida`` into ``("16", "ida")``; ``L35`` has no direction.

    The site is inconsistent about the summer line's slug, so ``l35`` and
    ``35-volta`` must end up in the same group.
    """
    for direction in ("ida", "volta"):
        if line_id.endswith(f"-{direction}"):
            slug = line_id[: -len(direction) - 1]
            return slug.lstrip("lL") or slug, direction
    return line_id.lstrip("lL") or line_id, ""


def busway_inventory(fetcher: Fetcher) -> list[LinePage]:
    """Every BusWay line page, from the operator's own sitemap."""
    sitemap = fetcher.get(BUSWAY_SITEMAP, BUSWAY, suffix=".xml")
    pages: list[LinePage] = []
    for url, line_id in _SITEMAP_LOC.findall(sitemap.read_text(errors="replace")):
        line, direction = parse_line_id(line_id)
        local = fetcher.get(url, BUSWAY)
        pages.append(LinePage(BUSWAY.key, line, direction, url, local))
    return sorted(
        pages,
        key=lambda p: (int(p.line) if p.line.isdigit() else 10**6, p.line, p.direction),
    )


def aveirobus_inventory(fetcher: Fetcher) -> list[LinePage]:
    """Every AveiroBus timetable PDF linked from its index pages.

    The link text is the line's own label (``1`` .. ``13``, ``azul``, ``verde``);
    the filename is only used for the raw artifact's name.
    """
    found: dict[str, LinePage] = {}
    for index in (AVEIROBUS_INDEX, *AVEIROBUS_EXTRA_INDEXES):
        page = fetcher.get(index, AVEIROBUS)
        html = page.read_text(errors="replace")
        for url, anchor in _AVEIROBUS_PDF.findall(html):
            if "/horarios/" not in url:
                continue
            label = re.sub(r"\s+", " ", _TAGS.sub(" ", anchor)).strip()
            local = fetcher.get(url, AVEIROBUS)
            found.setdefault(url, LinePage(AVEIROBUS.key, label or local.stem, "", url, local))
    return sorted(found.values(), key=lambda p: p.line)


def load_manifest() -> dict[str, dict[str, object]]:
    """The recorded URL -> checksum map."""
    if not paths.MANIFEST.exists():
        return {}
    return json.loads(paths.MANIFEST.read_text())
