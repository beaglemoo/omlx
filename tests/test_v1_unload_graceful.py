# SPDX-License-Identifier: Apache-2.0
"""POST /v1/models/{id}/unload drains through request_unload."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx import server
from omlx.exceptions import ModelBusyError, ModelLoadingError


def _pool(*, loaded=True, request_unload=True):
    entry = MagicMock()
    entry.engine = object() if loaded else None
    pool = MagicMock()
    pool.get_entry.return_value = entry
    pool.request_unload = AsyncMock(return_value=request_unload)
    pool._unload_engine = AsyncMock()
    return pool, entry


async def _call(pool, model_id="m"):
    with patch.object(server._server_state, "engine_pool", pool):
        return await server.unload_model(model_id, _=True)


@pytest.mark.asyncio
async def test_idle_model_unloads_and_keeps_response_shape():
    pool, _ = _pool(request_unload=True)
    assert await _call(pool) == {"status": "ok", "model_id": "m"}
    pool.request_unload.assert_awaited_once_with("m", reason="manual admin unload")
    pool._unload_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_model_waits_for_drain_then_reports_ok(monkeypatch):
    pool, entry = _pool(request_unload=False)
    calls = {"n": 0}

    def get_entry(_):
        calls["n"] += 1
        if calls["n"] > 3:
            entry.engine = None
        return entry

    pool.get_entry.side_effect = get_entry
    # First lookup is the 404/400 precheck; later ones are the drain poll.
    assert await _call(pool) == {"status": "ok", "model_id": "m"}
    pool._unload_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_still_draining_returns_202_unloading(monkeypatch):
    pool, _ = _pool(request_unload=False)
    monkeypatch.setattr(server, "_UNLOAD_DRAIN_WAIT_S", 0.05)
    response = await _call(pool)
    assert response.status_code == 202
    assert json.loads(response.body) == {
        "status": "unloading",
        "model_id": "m",
        "message": "Aborting active requests before unloading m",
    }


@pytest.mark.asyncio
async def test_loading_model_is_409():
    pool, _ = _pool()
    pool.request_unload.side_effect = ModelLoadingError("m", "still loading")
    with pytest.raises(HTTPException) as exc_info:
        await _call(pool)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_unload_busy_error_is_409():
    pool, _ = _pool()
    pool.request_unload.side_effect = ModelBusyError("m", "unload")
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
