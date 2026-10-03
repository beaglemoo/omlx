# SPDX-License-Identifier: Apache-2.0
"""Peer engine eviction (OMLX_PEER_EVICT_URLS) and its HTTP mapping."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from omlx import server
from omlx.engine_pool import EnginePool
from omlx.exceptions import InsufficientMemoryError, ModelTooLargeError, PeerBusyError

PEER = "http://127.0.0.1:8001"


class _Enforcer:
    """Dynamic-bound ceiling breakdown, like the real enforcer exposes."""

    memory_guard_tier = "balanced"

    def __init__(self, *, dynamic: int, static: int = 12_000, metal: int = 12_000):
        self.b = {"static": static, "dynamic": dynamic, "metal_cap": metal}

    def get_ceiling_breakdown(self):
        return dict(self.b)

    def wake(self, *, active: bool = False):
        return None


class _Peer:
    """Scriptable DwarfStar launcher served through httpx.MockTransport."""

    def __init__(self, *, status=None, stop_code=200, stop_unloads=True):
        self.status = {
            "loaded": True,
            "in_flight": 0,
            "uptime_seconds": 300.0,
            **(status or {}),
        }
        self.stop_code = stop_code
        self.stop_unloads = stop_unloads
        self.requests: list[httpx.Request] = []
        self.on_stop = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/admin/status":
            return httpx.Response(200, json=self.status)
        if request.url.path == "/admin/stop":
            if self.stop_code == 200 and self.stop_unloads:
                self.status["loaded"] = False
                if self.on_stop:
                    self.on_stop()
            return httpx.Response(self.stop_code, json={})
        return httpx.Response(404)

    @property
    def stops(self):
        return [r for r in self.requests if r.url.path == "/admin/stop"]


def _pool(tmp_path, monkeypatch, peer, *, dynamic=700, urls=PEER):
    monkeypatch.setenv("OMLX_PEER_EVICT_URLS", urls)
    model = tmp_path / "model-a"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model / "model.safetensors").write_bytes(b"0" * 1024)
    pool = EnginePool()
    enforcer = _Enforcer(dynamic=dynamic)
    pool._process_memory_enforcer = enforcer
    pool._get_final_ceiling = lambda: min(enforcer.b.values())
    pool._peer_evict_transport = httpx.MockTransport(peer.handler)
    pool.discover_models(str(tmp_path))
    return pool, enforcer


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr("omlx.engine_pool._ADMISSION_CEILING_RECHECK_S", 0)
    monkeypatch.setattr("omlx.engine_pool._PEER_UNLOAD_POLL_S", 0)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)


class _Admitted(Exception):
    pass


def test_urls_parsed_from_env(tmp_path, monkeypatch):
    pool, _ = _pool(
        tmp_path,
        monkeypatch,
        _Peer(),
        urls=" http://a:1/ , http://b:2,, ",
    )
    assert pool._peer_evict_urls == ["http://a:1", "http://b:2"]


def test_no_env_means_no_peers(monkeypatch):
    monkeypatch.delenv("OMLX_PEER_EVICT_URLS", raising=False)
    assert EnginePool()._peer_evict_urls == []


@pytest.mark.asyncio
async def test_idle_peer_is_stopped_with_if_idle_and_load_admitted(
    tmp_path, monkeypatch
):
    peer = _Peer()
    pool, enforcer = _pool(tmp_path, monkeypatch, peer)
    # The dynamic ceiling recovers once the peer's memory is released.
    peer.on_stop = lambda: enforcer.b.update(dynamic=12_000)

    with (
        patch.object(pool, "_load_engine", AsyncMock(side_effect=_Admitted)),
        pytest.raises(_Admitted),
    ):
        await pool.get_engine("model-a")

    assert len(peer.stops) == 1
    assert peer.stops[0].url.params["if_idle"] == "1"
    assert peer.stops[0].method == "POST"


@pytest.mark.asyncio
async def test_evicted_gets_five_second_recheck_budget(tmp_path, monkeypatch):
    """After an eviction the ceiling is re-read for about 5s (20 x 0.25s),
    even if it no longer reads as dynamic-bound."""
    peer = _Peer()
    pool, enforcer = _pool(tmp_path, monkeypatch, peer)
    reads = {"n": 0}

    def ceiling():
        reads["n"] += 1
        # Recovers only on the 10th read after the stop, beyond the normal
        # 4-recheck budget.
        return 12_000 if reads["n"] > 12 and peer.stops else 700

    pool._get_final_ceiling = ceiling
    enforcer.b["dynamic"] = 700

    with (
        patch.object(pool, "_load_engine", AsyncMock(side_effect=_Admitted)),
        pytest.raises(_Admitted),
    ):
        await pool.get_engine("model-a")
    assert len(peer.stops) == 1


@pytest.mark.asyncio
async def test_evicted_but_still_too_big_raises_memory_error(tmp_path, monkeypatch):
    peer = _Peer()
    pool, _ = _pool(tmp_path, monkeypatch, peer)
    with pytest.raises((InsufficientMemoryError, ModelTooLargeError)):
        await pool.get_engine("model-a")
    assert len(peer.stops) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        {"in_flight": 2},
        {"starting": True},
        {"starting": True, "loaded": False},
        {"uptime_seconds": 5.0},
    ],
    ids=["in_flight", "starting", "starting_not_loaded", "thrash_guard"],
)
async def test_busy_peer_raises_peer_busy_without_stopping(
    tmp_path, monkeypatch, status
):
    peer = _Peer(status=status)
    pool, _ = _pool(tmp_path, monkeypatch, peer)

    with pytest.raises(PeerBusyError) as exc_info:
        await pool.get_engine("model-a")

    assert peer.stops == []
    assert exc_info.value.retry_after_s == 15
    assert PEER in str(exc_info.value)


@pytest.mark.asyncio
async def test_stop_refused_with_409_is_busy(tmp_path, monkeypatch):
    peer = _Peer(stop_code=409)
    pool, _ = _pool(tmp_path, monkeypatch, peer)
    with pytest.raises(PeerBusyError):
        await pool.get_engine("model-a")
    assert len(peer.stops) == 1


@pytest.mark.asyncio
async def test_peer_not_loaded_is_none_and_no_stop(tmp_path, monkeypatch):
    peer = _Peer(status={"loaded": False})
    pool, _ = _pool(tmp_path, monkeypatch, peer)
    assert (await pool._evict_peers())[0] == "none"
    with pytest.raises((InsufficientMemoryError, ModelTooLargeError)):
        await pool.get_engine("model-a")
    assert peer.stops == []


@pytest.mark.asyncio
async def test_unreachable_peer_is_none(tmp_path, monkeypatch):
    peer = _Peer()
    pool, _ = _pool(tmp_path, monkeypatch, peer)

    def boom(request):
        raise httpx.ConnectError("refused")

    pool._peer_evict_transport = httpx.MockTransport(boom)
    assert (await pool._evict_peers())[0] == "none"
    with pytest.raises((InsufficientMemoryError, ModelTooLargeError)):
        await pool.get_engine("model-a")


@pytest.mark.asyncio
async def test_peer_that_never_unloads_is_busy(tmp_path, monkeypatch):
    peer = _Peer(stop_unloads=False)
    pool, _ = _pool(tmp_path, monkeypatch, peer)
    monkeypatch.setattr("omlx.engine_pool._PEER_STOP_TIMEOUT_S", 0.05)
    outcome, _, reason = await pool._evict_peers()
    assert (outcome, reason) == ("busy", "still unloading")


@pytest.mark.asyncio
async def test_static_bound_ceiling_never_contacts_peer(tmp_path, monkeypatch):
    peer = _Peer()
    pool, enforcer = _pool(tmp_path, monkeypatch, peer, dynamic=12_000)
    enforcer.b["static"] = 700
    with pytest.raises((InsufficientMemoryError, ModelTooLargeError)):
        await pool.get_engine("model-a")
    assert peer.requests == []


@pytest.mark.asyncio
async def test_no_peers_configured_never_contacts_anything(tmp_path, monkeypatch):
    peer = _Peer()
    pool, _ = _pool(tmp_path, monkeypatch, peer, urls="")
    with pytest.raises((InsufficientMemoryError, ModelTooLargeError)):
        await pool.get_engine("model-a")
    assert peer.requests == []


# --- HTTP mapping ---------------------------------------------------------


def test_peer_busy_maps_to_503_with_retry_after():
    exc = server._peer_busy_http_exception(
        PeerBusyError("model-a", PEER, "in_flight")
    )
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 503
    assert exc.headers == {"Retry-After": "15"}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/models/x/load"])
async def test_exception_handler_keeps_retry_after_header(path):
    request = MagicMock()
    request.url.path = path
    request.method = "POST"
    exc = server._peer_busy_http_exception(PeerBusyError("m", PEER, "in_flight"))

    response = await server.http_exception_handler(request, exc)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "15"
    body = json.loads(response.body)
    assert body["error"]["type"] == "server_error"
    assert "busy" in body["error"]["message"]


@pytest.mark.asyncio
async def test_get_engine_maps_peer_busy_to_503():
    pool = MagicMock()
    pool.get_engine = AsyncMock(side_effect=PeerBusyError("m", PEER, "in_flight"))
    pool.resolve_model_id.return_value = "m"
    state = server._server_state
    with (
        patch.object(server, "get_engine_pool", return_value=pool),
        patch.object(state, "settings_manager", None),
        patch.object(state, "global_settings", None),
        pytest.raises(HTTPException) as exc_info,
    ):
        await server.get_engine("m")
    assert exc_info.value.status_code == 503
    assert exc_info.value.headers["Retry-After"] == "15"


@pytest.mark.asyncio
async def test_load_endpoint_maps_peer_busy_to_503():
    entry = MagicMock(engine=None)
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.get_engine = AsyncMock(side_effect=PeerBusyError("m", PEER, "starting"))
    with (
        patch.object(server._server_state, "engine_pool", pool),
        pytest.raises(HTTPException) as exc_info,
    ):
        await server.load_model_public("m", _=True)
    assert exc_info.value.status_code == 503
    assert exc_info.value.headers["Retry-After"] == "15"
