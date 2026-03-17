from __future__ import annotations

import asyncio
import unittest

from app.event_hub import EventHub


class TestEventHub(unittest.IsolatedAsyncioTestCase):
    async def test_publish_reaches_subscriber(self) -> None:
        hub = EventHub()
        subscriber = hub.subscribe()

        next_item_task = asyncio.create_task(subscriber.__anext__())
        await asyncio.sleep(0)
        await hub.publish({"event_type": "status", "mode": "live"})

        item = await asyncio.wait_for(next_item_task, timeout=1)
        self.assertEqual(item["event_type"], "status")
        self.assertEqual(item["mode"], "live")

        await subscriber.aclose()


if __name__ == "__main__":
    unittest.main()
