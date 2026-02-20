from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SNAPSHOT_PATH = "/tmp/chainlink_btcusd.json"


@dataclass
class ChainlinkSnapshot:
    price: float
    updated_at: int
    fetched_at_ms: int
    round_id: int
    age_s: int | None


class ChainlinkSnapshotReader:
    """
    Strategy-friendly reader with tiny in-process cache.

    Usage from a strategy:
      reader = ChainlinkSnapshotReader()
      snap = reader.read(max_staleness_s=90)
      if snap:
          price = snap.price
    """

    def __init__(
        self, path: str = DEFAULT_SNAPSHOT_PATH, cache_ttl_ms: int = 150
    ) -> None:
        self.path = Path(path)
        self.cache_ttl_ms = max(0, cache_ttl_ms)
        self._last_read_ms = 0
        self._cached: ChainlinkSnapshot | None = None

    def _parse(self, data: dict[str, Any]) -> ChainlinkSnapshot | None:
        try:
            return ChainlinkSnapshot(
                price=float(data["price"]),
                updated_at=int(data["updated_at"]),
                fetched_at_ms=int(data["fetched_at_ms"]),
                round_id=int(data["round_id"]),
                age_s=(int(data["age_s"]) if data.get("age_s") is not None else None),
            )
        except Exception:
            return None

    def read(self, max_staleness_s: int | None = None) -> ChainlinkSnapshot | None:
        now_ms = int(time.time() * 1000)
        if self._cached and (now_ms - self._last_read_ms) <= self.cache_ttl_ms:
            snap = self._cached
        else:
            try:
                raw = self.path.read_text()
                data = json.loads(raw)
                snap = self._parse(data)
                self._cached = snap
                self._last_read_ms = now_ms
            except Exception:
                return None

        if snap is None:
            return None
        if max_staleness_s is not None:
            age = max(0, int(time.time()) - snap.updated_at)
            if age > max_staleness_s:
                return None
        return snap
