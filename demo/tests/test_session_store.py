from __future__ import annotations

import unittest

from app.session_store import SessionStore


class TestSessionStore(unittest.TestCase):
    def test_register_and_update_child(self) -> None:
        store = SessionStore()
        child = store.register_child("Kid 1", "192.168.1.10")
        updated = store.register_child("Kid Renamed", "192.168.1.10")

        self.assertEqual(child.child_id, updated.child_id)
        self.assertEqual(updated.codename, "Kid Renamed")

    def test_ingest_rssi_returns_window_at_10(self) -> None:
        store = SessionStore()
        child = store.register_child("Kid", "192.168.1.20")

        window = None
        for i in range(10):
            _, window = store.ingest_rssi(child.ip, -50 - i)

        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(len(window), 10)
        self.assertEqual(child.status, "predicting")

    def test_deterministic_replay_ip(self) -> None:
        store = SessionStore()
        self.assertEqual(
            store.deterministic_replay_ip("kid-a"),
            store.deterministic_replay_ip("kid-a"),
        )


if __name__ == "__main__":
    unittest.main()
