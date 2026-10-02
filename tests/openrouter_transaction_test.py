"""Ledger contention queues before SQLite while network calls stay concurrent."""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from exact.llm.ledger import RequestLedger


def test_short_sqlite_timeout_does_not_drop_queued_transaction(tmp_path, monkeypatch):
    ledger = RequestLedger(tmp_path)
    connect = sqlite3.connect

    def impatient(*args, **kwargs):
        kwargs["timeout"] = 0.02
        return connect(*args, **kwargs)

    monkeypatch.setattr("exact.llm.ledger.sqlite3.connect", impatient)
    started = Event()

    def contender():
        started.set()
        return RequestLedger(tmp_path).plan({"role": "decision", "payload": {"model": "fixture"}})

    with ThreadPoolExecutor(max_workers=1) as pool:
        with ledger._transaction():
            pending = pool.submit(contender)
            assert started.wait(2)
            time.sleep(0.15)
        assert len(pending.result(timeout=2)) == 64


def test_transaction_exception_rolls_back_and_releases_lock(tmp_path):
    ledger = RequestLedger(tmp_path)
    with pytest.raises(RuntimeError, match="fixture"):
        with ledger._transaction() as db:
            db.execute("INSERT INTO requests VALUES ('bad', '{}')")
            raise RuntimeError("fixture")
    with ledger._transaction() as db:
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0
    assert len(ledger.plan({"role": "decision"})) == 64
