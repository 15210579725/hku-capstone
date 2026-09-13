#!/usr/bin/env python3
"""
Persistent embedding cache (sqlite + float32 blobs).

Survives process exit, so re-runs and different k values never re-embed the
same sentence. Keyed by (model, dim, sha1(text)).
"""

import hashlib, sqlite3, threading
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / ".emb_cache" / "embeddings.sqlite"


class EmbeddingCache:
    def __init__(self, path=None, model="", dim=0):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.dim = dim
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS emb ("
            " key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        self._conn.commit()

    def _key(self, text):
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{self.model}|{self.dim}|{h}"

    def get_many(self, texts):
        """Return {text: np.ndarray} for texts present in the cache."""
        keys = {self._key(t): t for t in texts}
        out = {}
        items = list(keys.items())
        with self._lock:
            for i in range(0, len(items), 500):
                chunk = items[i:i + 500]
                qs = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT key, vec FROM emb WHERE key IN ({qs})",
                    [k for k, _ in chunk]).fetchall()
                by_key = dict(chunk)
                for k, blob in rows:
                    out[by_key[k]] = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
        return out

    def put_many(self, pairs):
        """pairs: iterable of (text, np.ndarray)."""
        rows = [(self._key(t), np.asarray(v, dtype=np.float32).tobytes())
                for t, v in pairs]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)", rows)
            self._conn.commit()

    def count(self):
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM emb").fetchone()[0]

    def close(self):
        with self._lock:
            self._conn.close()


if __name__ == "__main__":
    c = EmbeddingCache(model="test", dim=4)
    c.put_many([("hello", np.array([1, 2, 3, 4], dtype=np.float64))])
    print("count:", c.count())
    print("get:", c.get_many(["hello", "missing"]))
