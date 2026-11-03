"""재생 지연 때 참고용: 클라이언트(Tailscale IP)와의 경로가 직접인지 릴레이(DERP)인지 `tailscale status --json`으로 본다.
바이너리가 없거나 느리면 조용히 건너뛴다. 결과는 20초 캐시."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from typing import Optional

log = logging.getLogger("netpath")

_CANDIDATES = ("tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
               "/usr/local/bin/tailscale", "/opt/homebrew/bin/tailscale")
_cache: dict[str, tuple[float, str]] = {}
CACHE_S = 20.0


def _binary() -> Optional[str]:
    for c in _CANDIDATES:
        p = shutil.which(c) if "/" not in c else (c if shutil.os.path.exists(c) else None)
        if p:
            return p
    return None


async def peer_path(ip: str, timeout_s: float = 1.5) -> Optional[str]:
    """'direct 192.168.45.105:41641' / 'relay tok' / 'unknown'. 실패하면 None."""
    now = time.monotonic()
    hit = _cache.get(ip)
    if hit and now - hit[0] < CACHE_S:
        return hit[1]
    exe = _binary()
    if not exe:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(exe, "status", "--json", stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        d = json.loads(out.decode() or "{}")
    except Exception:  # noqa: BLE001 없거나 느리면 넘어감
        return None
    desc = "unknown"
    for p in (d.get("Peer") or {}).values():
        if ip in (p.get("TailscaleIPs") or []):
            cur, relay = p.get("CurAddr") or "", p.get("Relay") or ""
            desc = f"direct {cur}" if cur else (f"relay {relay}" if relay else "no path")
            break
    _cache[ip] = (now, desc)
    return desc


async def nudge(ip: str, timeout_s: float = 2.0) -> Optional[str]:
    """`tailscale ping -c 1`로 직접 경로 탐색을 재촉한다(릴레이로 떨어진 뒤 복귀를 앞당김). 결과 한 줄 또는 None."""
    exe = _binary()
    if not exe:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(exe, "ping", "-c", "1", "--timeout", "1s", ip,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        line = (out.decode(errors="ignore").strip().splitlines() or [""])[-1]
        _cache.pop(ip, None)
        return line[:120]
    except Exception:  # noqa: BLE001
        return None
