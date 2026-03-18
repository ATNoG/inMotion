from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import DemoConfig
from app.scanner_adapter import ScannerAdapter


def make_config(base: Path, csv_path: Path) -> DemoConfig:
    return DemoConfig(
        root_dir=base,
        model_path=base / "models" / "dummy.joblib",
        fallback_csv_path=csv_path,
        test_router_csv_path=csv_path,
        scan_interval_seconds=0.01,
        scan_mode="auto",
        router_ip="",
        router_user="root",
        router_pass="",
        router_key_file="",
        router_port=22,
        router_command_mode="auto",
    )


class TestScannerAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_tick_replay_emits_rssi_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            csv_path = base / "sample.csv"
            csv_path.write_text(
                "ip,1,2,3,4,5,6,7,8,9,10,label\n"
                "aa:aa:aa:aa:aa:aa,-40,-41,-42,-43,-44,-45,-46,-47,-48,-49,AA\n"
            )

            adapter = ScannerAdapter(make_config(base, csv_path))
            emitted = []

            async def on_rssi(event):
                emitted.append(event)

            await adapter._tick_replay(on_rssi, lambda: ["192.168.1.20"])
            await adapter._tick_replay(on_rssi, lambda: ["192.168.1.20"])

            self.assertEqual(len(emitted), 2)
            self.assertEqual(emitted[0].source, "replay")
            self.assertTrue(-60 <= emitted[0].rssi <= -30)

    async def test_detect_unassigned_ip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            csv_path = base / "sample.csv"
            csv_path.write_text(
                "ip,1,2,3,4,5,6,7,8,9,10,label\n"
                "bb:bb:bb:bb:bb:bb,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,BB\n"
            )

            adapter = ScannerAdapter(make_config(base, csv_path))

            async def on_rssi(_event):
                return None

            await adapter._tick_replay(on_rssi, lambda: ["192.168.1.30"])

            found = adapter.detect_unassigned_ip(set())
            self.assertEqual(found, "192.168.1.30")
            self.assertIsNone(adapter.detect_unassigned_ip({"192.168.1.30"}))


if __name__ == "__main__":
    unittest.main()
