"""Process-owned, bounded SQLite access for exact-text encoder vectors."""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import torch


class EncoderStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, shape TEXT, raw BLOB, sha256 TEXT)")
        self.counts = dict(connections=1, lookups=0, transactions=0, hits=0,
                           misses=0, corrupt=0, bytes_read=0, bytes_written=0,
                           skipped_writes=0)

    def lookup(self, keys, dtype, dimension=None):
        result = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), 400):
            block = unique[start:start + 400]
            self.counts["lookups"] += 1
            query = "SELECT key,shape,raw,sha256 FROM vectors WHERE key IN (" + ",".join("?" for _ in block) + ")"
            for key, shape, raw, digest in self.db.execute(query, block):
                try:
                    metadata = json.loads(shape)
                    # Legacy dtype is bound by its encoder key; new rows make it explicit.
                    if isinstance(metadata, dict):
                        if metadata["dtype"] != str(dtype):
                            raise ValueError("dtype")
                        shape = metadata["shape"]
                    else:
                        shape = metadata
                    if (not isinstance(shape, list) or len(shape) != 1
                            or not isinstance(shape[0], int) or shape[0] <= 0
                            or (dimension is not None and shape[0] != dimension)
                            or len(raw) != math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                            or hashlib.sha256(raw).hexdigest() != digest):
                        raise ValueError("shape or checksum")
                    result[key] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
                    self.counts["bytes_read"] += len(raw)
                except (ValueError, TypeError, KeyError, RuntimeError):
                    self.counts["corrupt"] += 1
        self.counts["hits"] += len(result)
        self.counts["misses"] += len(unique) - len(result)
        return result

    def publish(self, rows):
        if not rows:
            return
        encoded = []
        for key, row in rows.items():
            row = row.detach().cpu().contiguous()
            raw = row.view(torch.uint8).numpy().tobytes()
            encoded.append((key, json.dumps({"shape": list(row.shape), "dtype": str(row.dtype)}),
                            raw, hashlib.sha256(raw).hexdigest()))
        size = sum(len(row[2]) for row in encoded)
        limit = max(0, int(os.getenv("EXACT_EMBEDDING_CACHE_MAX_BYTES", str(8 * 1024**3))))
        growth = 2 * sum(len(raw) + len(key.encode()) + len(shape.encode()) + 512
                         for key, shape, raw, _ in encoded) + 65536
        try:
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                # Quota admission shares the writer lock: another process cannot
                # independently reserve the same physical DB/WAL headroom.
                physical = sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*"))
                if physical + growth > limit:
                    self.counts["skipped_writes"] += len(encoded)
                    return
                self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)", encoded)
            self.counts["transactions"] += 1
            self.counts["bytes_written"] += size
        except sqlite3.Error:
            self.db.rollback()
            self.counts["skipped_writes"] += len(encoded)

    def close(self):
        self.db.close()


_STORES = OrderedDict()


def close_encoder_stores():
    for store in _STORES.values():
        try:
            store.close()
        except sqlite3.Error:
            pass
    _STORES.clear()


def encoder_store(path):
    key = (os.getpid(), str(Path(path).resolve()))
    if any(pid != os.getpid() for pid, _ in _STORES):
        close_encoder_stores()
    if key not in _STORES:
        while len(_STORES) >= 4:
            _, old = _STORES.popitem(last=False)
            old.close()
        _STORES[key] = EncoderStore(path)
    _STORES.move_to_end(key)
    return _STORES[key]


def encoder_cache_stats():
    return {str(store.path): dict(store.counts) for store in _STORES.values()}


atexit.register(close_encoder_stores)
