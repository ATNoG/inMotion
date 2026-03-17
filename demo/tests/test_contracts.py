from __future__ import annotations

import unittest
from datetime import UTC, datetime

from app.contracts import StatusEvent


class TestContracts(unittest.TestCase):
    def test_status_event_accepts_replay_test_mode(self) -> None:
        event = StatusEvent(
            timestamp=datetime.now(UTC),
            status="start",
            mode="replay-test",
            message="ok",
        )
        self.assertEqual(event.mode, "replay-test")


if __name__ == "__main__":
    unittest.main()
