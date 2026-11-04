"""재생 지연 때 참고용: 클라이언트(Tailscale IP)와의 경로가 직접인지 릴레이(DERP)인지 `tailscale status --json`으로 본다.
바이너리가 없거나 느리면 건너뛴다(첫 실패 이유만 INFO로 한 번 남김). 결과는 20초 캐시."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from typing import Optional

log = logging.getLogger("netpath")

_CANDIDATES = ("tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
               "/usr/local/bin/tailscale", "/opt/homebrew/bin/tailscale")
_cache: dict[str, tuple[float, str]] = {}
CACHE_S = 20.0
_warned: set[str] = set()


def _binary() -> Optional[str]:
    for c in _CANDIDATES:
        p = shutil.which(c) if "/" not in c else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


def _env() -> dict[str, str]:
    # macOS Tailscale 앱 바이너리는 TERM이 없으면(launchd 최소 환경) CLI가 아니라 GUI를 띄우려다 실패한다.
    return {**os.environ, "TERM": os.environ.get("TERM") or "xterm"}


def _note_fail(kind: str, why: str) -> None:
    """종류별 첫 실패 이유만 INFO로 한 번 남긴다(매번 찍으면 로그가 넘침)."""
    if kind in _warned:
        return
    _warned.add(kind)
    log.info("tailscale %s 실패(이후 같은 실패는 생략): %s", kind, why[:200])


async def _run(exe: str, args: tuple[str, ...], timeout_s: float, merge_err: bool) -> tuple[int, bytes, bytes]:
    """하위 프로세스를 돌리고 (returncode, stdout, stderr). 시간 초과·취소면 프로세스를 죽이고 예외를 다시 던진다."""
    proc = await asyncio.create_subprocess_exec(
        exe, *args, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT if merge_err else asyncio.subprocess.PIPE, env=_env())
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except BaseException:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.shield(proc.wait())
            except BaseException:  # noqa: BLE001 정리 중 취소는 무시
                pass
        raise
    return proc.returncode or 0, out or b"", err or b""


async def peer_path(ip: str, timeout_s: float = 1.5) -> Optional[str]:
    """'direct 192.168.45.105:41641' / 'relay tok' / 'unknown'. 실패하면 None."""
    now = time.monotonic()
    hit = _cache.get(ip)
    if hit and now - hit[0] < CACHE_S:
        return hit[1]
    exe = _binary()
    if not exe:
        _note_fail("status", "바이너리 없음")
        return None
    try:
        rc, out, err = await _run(exe, ("status", "--json"), timeout_s, merge_err=False)
        if rc != 0:
            _note_fail("status", f"rc={rc} {err.decode(errors='ignore').strip()}")
            return None
        d = json.loads(out.decode() or "{}")
    except asyncio.TimeoutError:
        _note_fail("status", f"{timeout_s}s 시간 초과")
        return None
    except Exception as e:  # noqa: BLE001 없거나 느리면 넘어감
        _note_fail("status", repr(e))
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
        _note_fail("ping", "바이너리 없음")
        return None
    try:
        _rc, out, _ = await _run(exe, ("ping", "-c", "1", "--timeout", "1s", ip), timeout_s, merge_err=True)
        line = (out.decode(errors="ignore").strip().splitlines() or [""])[-1]
        _cache.pop(ip, None)
        return line[:120]
    except asyncio.TimeoutError:
        _note_fail("ping", f"{timeout_s}s 시간 초과")
        return None
    except Exception as e:  # noqa: BLE001
        _note_fail("ping", repr(e))
        return None
