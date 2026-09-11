"""Regenerate tests/golden/model_outputs.json.

Run deliberately:  uv run python tests/_regen_golden.py

Only do this when a model's output is *expected* to change. The diff is then part
of the review, and the commit message should say why.
"""

from __future__ import annotations

import json
import sys

from test_model_outputs_golden import GOLDEN_FILE, _fingerprint

from inmotion.models import registry as reg


def main() -> int:
    digests = {name: _fingerprint(name) for name in reg.spec_names()}
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_FILE} ({len(digests)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
