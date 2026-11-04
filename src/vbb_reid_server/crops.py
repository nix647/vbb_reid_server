"""Where a sighting's crop images come from.

1. LAN copy of the board's sighting store (``scripts/rsync_board_crops.sh``):
   ``<lan_root>/<did>/<YYYY-MM-DD>/<sighting_id>/<idx>.jpg`` — every crop the board
   kept (K=2 since 2026-10-19, up to 5 before), but only the last 7 days survive on
   the board, so copies must be pulled weekly.
2. The CDN best crop (``image_url``, ``{sighting_id}.jpg``), cached under
   ``<cdn_cache>/<did>/<day>/<sighting_id>.jpg``.

The LAN set wins when present: it contains the uploaded best crop too (the board
uploads one of its K crops). Crops are inputs only; nothing here keeps them past
``prune()`` (decision 14: no crop retention on GPU hosts).
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .data import Sighting


@dataclass(frozen=True)
class CropRef:
    sighting_id: str
    index: int  # 0..K-1 for LAN crops; 0 for the CDN crop
    path: Path
    source: str  # "lan" | "cdn"


class CropStore:
    def __init__(self, lan_root: str | Path | None = None, cdn_cache: str | Path | None = None,
                 *, fetch: bool = True, timeout_s: float = 15.0, retries: int = 2,
                 user_agent: str = "vbb-reid-server/0.1"):
        self.lan_root = Path(lan_root) if lan_root else None
        self.cdn_cache = Path(cdn_cache) if cdn_cache else None
        self.fetch = fetch
        self.timeout_s = timeout_s
        self.retries = retries
        self.user_agent = user_agent
        self.stats = {"lan": 0, "cdn_hit": 0, "cdn_fetched": 0, "cdn_failed": 0, "missing": 0}
        self.fetched: list[Path] = []  # crops this store downloaded (a production run deletes them afterwards)

    # ------------------------------------------------------------- lookup --
    def lan_crops(self, s: Sighting) -> list[CropRef]:
        if self.lan_root is None:
            return []
        out: list[CropRef] = []
        # Layout written by rsync_board_crops.sh: <did>/<day>/<sid>/<idx>.jpg. The board's
        # own day folder is its LOCAL calendar day, which equals the facility day for
        # the test boards (both Asia/Shanghai); also look one day either side.
        for day in _around(s.day):
            d = self.lan_root / s.did / day / s.sighting_id
            if d.is_dir():
                for f in sorted(d.glob("*.jpg")):
                    stem = f.stem
                    if stem.isdigit():
                        out.append(CropRef(s.sighting_id, int(stem), f, "lan"))
                if out:
                    break
        return out

    def cdn_crop(self, s: Sighting) -> CropRef | None:
        if not s.image_url or self.cdn_cache is None:
            return None
        dest = self.cdn_cache / s.did / s.day / f"{s.sighting_id}.jpg"
        if dest.is_file() and dest.stat().st_size > 0:
            self.stats["cdn_hit"] += 1
            return CropRef(s.sighting_id, 0, dest, "cdn")
        if not self.fetch:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(s.image_url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp, tmp.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
                if tmp.stat().st_size == 0:
                    raise OSError("empty body")
                os.replace(tmp, dest)
                self.stats["cdn_fetched"] += 1
                self.fetched.append(dest)
                return CropRef(s.sighting_id, 0, dest, "cdn")
            except Exception:
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        self.stats["cdn_failed"] += 1
        return None

    def crops(self, s: Sighting) -> list[CropRef]:
        lan = self.lan_crops(s)
        if lan:
            self.stats["lan"] += 1
            return lan
        c = self.cdn_crop(s)
        if c is None:
            self.stats["missing"] += 1
            return []
        return [c]

    def delete_fetched(self) -> int:
        """Delete every crop this store downloaded. Returns files removed."""
        n = 0
        for p in self.fetched:
            try:
                p.unlink()
                n += 1
            except FileNotFoundError:
                pass
        self.fetched = []
        return n

    # -------------------------------------------------------------- prune --
    def prune_cdn(self, older_than_days: float) -> int:
        """Delete cached CDN crops older than N days (by mtime). Returns files removed."""
        if self.cdn_cache is None or not self.cdn_cache.exists():
            return 0
        cutoff = time.time() - older_than_days * 86400
        n = 0
        for f in self.cdn_cache.rglob("*.jpg"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                n += 1
        return n


def _around(day: str) -> list[str]:
    from datetime import date, timedelta
    d = date.fromisoformat(day)
    return [day, (d - timedelta(days=1)).isoformat(), (d + timedelta(days=1)).isoformat()]
