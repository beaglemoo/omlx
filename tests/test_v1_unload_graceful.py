# SPDX-License-Identifier: Apache-2.0
"""POST /v1/models/{id}/unload: refuse while serving, ?force=1 aborts."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx import server
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import ModelBusyError, ModelLoadingError


def _pool(*, loaded=True, request_unload=True, unload_if_idle=True):
    entry = MagicMock()
    entry.engine = object() if loaded else None
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=request_unload)
    pool.unload_if_idle = AsyncMock(return_value=unload_if_idle)
    pool._unload_engine = AsyncMock()
    return pool, entry


async def _call(pool, model_id="m", **kwargs):
    with patch.object(server._server_state, "engine_pool", pool):
        return await server.unload_model(model_id, _=True, **kwargs)


# --- default path: refuse while serving ------------------------------------


@pytest.mark.asyncio
async def test_idle_model_unloads_without_abort():
    pool, _ = _pool(unload_if_idle=True)
    assert await _call(pool) == {"status": "ok", "model_id": "m"}
    pool.unload_if_idle.assert_awaited_once_with("m")
    pool.request_unload.assert_not_awaited()
    pool._unload_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_model_finishing_within_window_unloads_200(monkeypatch):
    pool, _ = _pool()
    pool.unload_if_idle.side_effect = [False, False, True]
    monkeypatch.setattr(server, "_UNLOAD_DRAIN_WAIT_S", 5.0)
    assert await _call(pool) == {"status": "ok", "model_id": "m"}
    assert pool.unload_if_idle.await_count == 3
    pool.request_unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_model_past_window_is_409_model_busy_and_not_aborted(monkeypatch):
    pool, _ = _pool(unload_if_idle=False)
    monkeypatch.setattr(server, "_UNLOAD_DRAIN_WAIT_S", 0.05)
    response = await _call(pool)
    assert response.status_code == 409
    error = json.loads(response.body)["error"]
    assert error["type"] == "model_busy"
    assert error["model_id"] == "m"
    # Nothing was aborted, queued or torn down.
    pool.request_unload.assert_not_awaited()
    pool._unload_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_loading_model_is_409_on_default_path():
    pool, _ = _pool()
    pool.unload_if_idle.side_effect = ModelLoadingError("m", "still loading")
    with pytest.raises(HTTPException) as exc_info:
        await _call(pool)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_teardown_in_progress_is_409_on_default_path():
    pool, _ = _pool()
    pool.unload_if_idle.side_effect = ModelBusyError("m", "unload")
    with pytest.raises(HTTPException) as exc_info:
        await _call(pool)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_unknown_and_unloaded_models_keep_status_codes():
    pool, _ = _pool()
    pool.get_entry.return_value = None
    with pytest.raises(HTTPException) as exc_info:
        await _call(pool)
    assert exc_info.value.status_code == 404

    pool, _ = _pool(loaded=False)
    with pytest.raises(HTTPException) as exc_info:
        await _call(pool)
    assert exc_info.value.status_code == 400
    pool.unload_if_idle.assert_not_awaited()


# --- ?force=1: abort, drain, unload (the pre-fix behaviour) -----------------


@pytest.mark.asyncio
async def test_force_idle_model_unloads_through_request_unload():
    pool, _ = _pool(request_unload=True)
    assert await _call(pool, force=True) == {"status": "ok", "model_id": "m"}
    pool.request_unload.assert_awaited_once_with("m", reason="manual admin unload")
    pool.unload_if_idle.assert_not_awaited()


@pytest.mark.asyncio
async def test_force_waits_for_abort_drain_then_reports_ok():
    pool, entry = _pool(request_unload=False)
    calls = {"n": 0}

    def get_entry(_):
        calls["n"] += 1
        if calls["n"] > 3:
            entry.engine = None
        return entry

    pool.get_entry.side_effect = get_entry
    assert await _call(pool, force=True) == {"status": "ok", "model_id": "m"}
    pool._unload_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_force_still_draining_returns_202_unloading(monkeypatch):
    pool, _ = _pool(request_unload=False)
    monkeypatch.setattr(server, "_UNLOAD_DRAIN_WAIT_S", 0.05)
    response = await _call(pool, force=True)
    assert response.status_code == 202
    assert json.loads(response.body) == {
        "status": "unloading",
        "model_id": "m",
        "message": "Aborting active requests before unloading m",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ModelLoadingError("m", "still loading"), ModelBusyError("m", "unload")]
)
async def test_force_pool_refusals_are_409(error):
    pool, _ = _pool()
    pool.request_unload.side_effect = error
    with pytest.raises(HTTPException) as exc_info:
        await _call(pool, force=True)
    assert exc_info.value.status_code == 409


# --- EnginePool.unload_if_idle with a real pool -----------------------------


def _real_pool(*, busy=False, in_use=0):
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    engine = MagicMock()
    engine.has_active_requests.return_value = busy
    engine.abort_all_requests = AsyncMock()
    engine.stop = AsyncMock()
    engine.scheduler = None
    engine._engine = None
    entry = EngineEntry(
        model_id="m",
        model_path="/models/m",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
        engine=engine,
        in_use=in_use,
    )
    pool._entries = {"m": entry}
    pool._unload_engine = AsyncMock()
    return pool, entry, engine


@pytest.mark.asyncio
async def test_pool_unload_if_idle_unloads_idle_model():
    pool, _, _ = _real_pool()
    assert await pool.unload_if_idle("m") is True
    pool._unload_engine.assert_awaited_once_with("m")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["active_request", "lease"])
async def test_pool_unload_if_idle_leaves_busy_model_untouched(kind):
    pool, entry, engine = _real_pool(
        busy=kind == "active_request", in_use=1 if kind == "lease" else 0
    )
    assert await pool.unload_if_idle("m") is False
    pool._unload_engine.assert_not_awaited()
    engine.abort_all_requests.assert_not_awaited()
    assert entry.engine is engine
    assert not entry.pending_unload_reason
    assert not entry.abort_requested
    assert not pool._pending_unload_tasks


@pytest.mark.asyncio
async def test_pool_unload_if_idle_missing_or_unloaded_is_true():
    pool, entry, _ = _real_pool()
    entry.engine = None
    assert await pool.unload_if_idle("m") is True
    assert await pool.unload_if_idle("nope") is True
    pool._unload_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_pool_unload_if_idle_refuses_loading_model():
    pool, entry, _ = _real_pool()
    entry.is_loading = True
    with pytest.raises(ModelLoadingError):
        await pool.unload_if_idle("m")


@pytest.mark.asyncio
async def test_in_flight_request_survives_refused_unload_then_unloads(monkeypatch):
    """End to end through the route: a live lease keeps the stream alive."""
    pool, entry, engine = _real_pool(in_use=1)
    monkeypatch.setattr(server, "_UNLOAD_DRAIN_WAIT_S", 0.3)

    refused = await _call(pool)
    assert refused.status_code == 409
    pool._unload_engine.assert_not_awaited()
    engine.abort_all_requests.assert_not_awaited()
    assert entry.engine is engine and entry.in_use == 1

    async def finish_soon():
        await asyncio.sleep(0.1)
        entry.in_use = 0

    monkeypatch.setattr(server, "_UNLOAD_DRAIN_WAIT_S", 5.0)
    finisher = asyncio.create_task(finish_soon())
    assert await _call(pool) == {"status": "ok", "model_id": "m"}
    await finisher
    pool._unload_engine.assert_awaited_once_with("m")
    engine.abort_all_requests.assert_not_awaited()
