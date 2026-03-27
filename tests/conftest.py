"""Shared pytest fixtures."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
import aiosqlite

from src.history.db import init_db


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture
async def tmp_db():
    """In-memory SQLite DB for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = await init_db(db_path)
        yield conn
        await conn.close()
