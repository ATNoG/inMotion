"""Regenerate tests/golden/model_outputs.json.

Run deliberately:  uv run python tests/_regen_golden.py

Only do this when a model's output is *expected* to change. The diff is then part
of the review, and the commit message should say why. A drift that is only kernel
noise does not need it: the comparison in test_model_outputs_golden.py is a
tolerance precisely so that rebuilding the environment does not require this.
"""

from __future__ import annotations

import json
import sys

from test_model_outputs_golden import GOLDEN_FILE, _output

from inmotion.models import registry as reg


def main() -> int:
    outputs = {name: _output(name).tolist() for name in reg.spec_names()}
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_FILE} ({len(outputs)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
