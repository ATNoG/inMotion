"""Flatten a golden.json into the TSV the host parity test reads.

The C harness deliberately does not parse JSON: a hand-rolled parser is one more
thing that can disagree with the fixture, and it would make a parity FAIL
ambiguous between a maths bug and a parsing bug. This does the parsing once, in
Python, where the fixture is already valid.

Usage:
    uv run python firmware/tools/dump_cases.py /tmp/lejepa_int8/golden.json /tmp/lejepa_int8/cases.tsv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} golden.json cases.tsv")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    payload = json.loads(src.read_text())
    lines = [
        f"# seq_len={payload['seq_len']} in_channels={payload['in_channels']} "
        f"num_classes={payload['num_classes']} tol={payload['tol']}"
    ]
    for case in payload["cases"]:
        flat = [v for row in case["input"] for v in row]
        lines.append(
            ",".join(repr(v) for v in flat)
            + "\t"
            + ",".join(repr(v) for v in case["logits"])
        )
    dst.write_text("\n".join(lines) + "\n")
    print(f"wrote {dst}: {len(payload['cases'])} cases, {payload['seq_len']}x"
          f"{payload['in_channels']} inputs")


if __name__ == "__main__":
    main()
