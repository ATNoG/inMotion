from __future__ import annotations

import asyncio
import contextlib
import csv
import importlib.util
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .config import DemoConfig


@dataclass
class ScannerEvent:
    timestamp: datetime
    ip: str
    rssi: float
    source: Literal["live", "replay"]


class ScannerAdapter:
    def __init__(self, config: DemoConfig) -> None:
        self._config = config
        self._running = False
        self._task: asyncio.Task | None = None
        self._scanner = None
        self._mode: str = "live"
        self._latest_seen: dict[str, datetime] = {}
        self._fallback_replay_windows = self._load_replay_windows(config.fallback_csv_path)
        self._test_replay_windows = self._load_replay_windows(config.test_router_csv_path)
        self._replay_state_by_ip: dict[str, tuple[list[float], int]] = {}
        self._reconnect_counter = 0
        self._live_ready_sent = False
        self._replay_ready_sent = False
        self._force_test_mode = False

    @property
    def mode(self) -> str:
        if self._force_test_mode:
            return "replay-test"
        return self._mode

    def set_test_mode(self, enabled: bool) -> None:
        self._force_test_mode = enabled
        if enabled:
            self._mode = "replay"

    def get_detected_ips(self) -> list[str]:
        return [
            ip
            for ip, _ in sorted(
                self._latest_seen.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    def detect_unassigned_ip(self, assigned_ips: set[str]) -> str | None:
        for ip in self.get_detected_ips():
            if ip not in assigned_ips:
                return ip
        return None

    async def start(
        self,
        on_rssi_event: Callable[[ScannerEvent], Awaitable[None] | None],
        on_status_event: Callable[[dict], Awaitable[None] | None],
        get_target_ips: Callable[[], list[str]],
    ) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(on_rssi_event, on_status_event, get_target_ips)
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run_loop(
        self,
        on_rssi_event: Callable[[ScannerEvent], Awaitable[None] | None],
        on_status_event: Callable[[dict], Awaitable[None] | None],
        get_target_ips: Callable[[], list[str]],
    ) -> None:
        await self._emit_status(on_status_event, "connect", "live", "A ligar ao scanner")

        while self._running:
            if self._force_test_mode:
                await self._tick_replay(on_rssi_event, get_target_ips)
                if not self._replay_ready_sent:
                    await self._emit_status(
                        on_status_event,
                        "start",
                        "replay",
                        "Modo de teste do router ativo",
                    )
                    self._replay_ready_sent = True
                    self._live_ready_sent = False
                await asyncio.sleep(self._config.scan_interval_seconds)
                continue

            if self._mode == "live":
                try:
                    await self._tick_live(on_rssi_event, get_target_ips)
                    if not self._live_ready_sent:
                        await self._emit_status(
                            on_status_event,
                            "start",
                            "live",
                            "Scanner ao vivo ativo",
                        )
                        self._live_ready_sent = True
                        self._replay_ready_sent = False
                except Exception as exc:
                    await self._emit_status(on_status_event, "error", "live", str(exc))
                    self._mode = "replay"
                    self._reconnect_counter = 0
                await asyncio.sleep(self._config.scan_interval_seconds)
            else:
                await self._tick_replay(on_rssi_event, get_target_ips)
                if not self._replay_ready_sent:
                    await self._emit_status(
                        on_status_event,
                        "start",
                        "replay",
                        "Fallback por replay ativo",
                    )
                    self._replay_ready_sent = True
                    self._live_ready_sent = False
                await asyncio.sleep(self._config.scan_interval_seconds)
                self._reconnect_counter += 1
                if self._reconnect_counter >= 15:
                    self._reconnect_counter = 0
                    try:
                        await self._tick_live(on_rssi_event, get_target_ips)
                        self._mode = "live"
                        await self._emit_status(
                            on_status_event,
                            "recover",
                            "live",
                            "Scanner ao vivo recuperado",
                        )
                    except Exception:
                        pass

    async def _tick_live(
        self,
        on_rssi_event: Callable[[ScannerEvent], Awaitable[None] | None],
        get_target_ips: Callable[[], list[str]],
    ) -> None:
        scanner = await self._ensure_live_scanner()
        all_clients = await asyncio.to_thread(scanner.get_all_wifi_clients, False)
        if not isinstance(all_clients, dict):
            raise RuntimeError("Unexpected scanner output")

        target_ips = {ip.lower() for ip in get_target_ips()}
        aggregated: dict[str, list[float]] = {}

        for clients in all_clients.values():
            if not isinstance(clients, list):
                continue
            for client in clients:
                ip = str(client.get("ip", "")).lower().strip()
                rssi_raw = client.get("rssi")
                if not ip:
                    continue
                try:
                    rssi = float(rssi_raw)
                except (TypeError, ValueError):
                    continue
                self._latest_seen[ip] = datetime.now(UTC)
                aggregated.setdefault(ip, []).append(rssi)

        for ip, values in aggregated.items():
            if target_ips and ip not in target_ips:
                continue
            avg_rssi = float(sum(values) / len(values))
            event = ScannerEvent(
                timestamp=datetime.now(UTC),
                ip=ip,
                rssi=avg_rssi,
                source="live",
            )
            maybe_task = on_rssi_event(event)
            if asyncio.iscoroutine(maybe_task):
                await maybe_task

    async def _tick_replay(
        self,
        on_rssi_event: Callable[[ScannerEvent], Awaitable[None] | None],
        get_target_ips: Callable[[], list[str]],
    ) -> None:
        target_ips = get_target_ips()
        if not target_ips:
            return

        for ip in target_ips:
            windows = (
                self._test_replay_windows
                if self._force_test_mode
                else self._fallback_replay_windows
            )
            active_window, idx = self._replay_state_by_ip.get(ip, ([], 0))
            if not active_window or idx >= len(active_window):
                active_window = random.choice(windows)
                idx = 0

            value = active_window[idx]
            self._replay_state_by_ip[ip] = (active_window, idx + 1)
            self._latest_seen[ip] = datetime.now(UTC)
            event = ScannerEvent(
                timestamp=datetime.now(UTC),
                ip=ip,
                rssi=value,
                source="replay",
            )
            maybe_task = on_rssi_event(event)
            if asyncio.iscoroutine(maybe_task):
                await maybe_task

    async def _ensure_live_scanner(self):
        if self._scanner is not None:
            return self._scanner
        if not self._config.router_ip:
            raise RuntimeError("Router não configurado; a usar fallback por replay")

        scanner_script = self._config.root_dir / "wavecom_files" / "wifi_scan.py"
        if not scanner_script.exists():
            raise RuntimeError("wifi_scan.py não encontrado")

        spec = importlib.util.spec_from_file_location("wifi_scan", scanner_script)
        if spec is None or spec.loader is None:
            raise RuntimeError("Não foi possível importar o módulo de scanner WiFi")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scanner_cls = getattr(module, "OpenWrtWiFiScanner", None)
        if scanner_cls is None:
            raise RuntimeError("Classe OpenWrtWiFiScanner em falta")

        self._scanner = scanner_cls(
            router_ip=self._config.router_ip,
            username=self._config.router_user,
            password=self._config.router_pass or None,
            key_file=self._config.router_key_file or None,
            port=self._config.router_port,
            mode=self._config.router_command_mode,
        )
        return self._scanner

    def _load_replay_windows(self, csv_path: Path) -> list[list[float]]:
        default_values = [[-52.0, -53.0, -55.0, -57.0, -54.0, -50.0, -48.0, -49.0, -51.0, -53.0]]
        if not csv_path.exists():
            return default_values

        windows: list[list[float]] = []
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                window: list[float] = []
                for i in range(1, 11):
                    key = str(i)
                    if key not in row:
                        continue
                    try:
                        window.append(float(row[key]))
                    except (TypeError, ValueError):
                        continue
                if len(window) == 10:
                    windows.append(window)
                if len(windows) >= 2000:
                    break

        return windows if windows else default_values

    async def _emit_status(
        self,
        callback: Callable[[dict], Awaitable[None] | None],
        status: str,
        mode: str,
        message: str,
    ) -> None:
        payload = {
            "event_type": "status",
            "timestamp": datetime.now(UTC),
            "status": status,
            "mode": mode,
            "message": message,
        }
        maybe_task = callback(payload)
        if asyncio.iscoroutine(maybe_task):
            await maybe_task
