"""Bus-network data: the Aveiro operators, and the OD matrices built from them.

Two operators cover the Aveiro region and neither publishes a machine-readable
feed, so both networks are scraped from the operator sites:

``busway``
    BusWay (CIRA), the intermunicipal network. Every line page renders its
    timetables as HTML tables of stops against trips.
``aveirobus``
    AveiroBus, the urban network of the city of Aveiro. Every line is a
    one-page PDF whose timetable is a single embedded raster image.

Submodules:

``paths``     where the raw artifacts and the derived tables live
``sources``   the polite fetch layer (robots-aware, cached, checksummed)
``busway``    HTML timetable tables to stop-time records
``aveirobus`` PDF timetables to stop-time records
``od``        the tidy trip/stop tables and the origin-destination matrices

The raw artifacts are a cache, not a deliverable: ``data/bus/raw`` is
gitignored and ``manifest.json`` records the URL and checksum of every file, so
the scrape can be replayed against the live sites.
"""

from __future__ import annotations

__all__ = ["Operator", "busway", "paths", "sources"]
