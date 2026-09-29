"""Model-independent, resumable storage for tidy predictions."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time

import pandas as pd

from .core import normalize_prediction_table, variant_key
from .model_base import VariantRecord


class PredictionStore:
    """SQLite prediction sink shared by all model adapters.

    A completed variant may legitimately have no prediction rows (for example,
    no gene lies in the model window), so completion is tracked separately.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 60.0
        while True:
            connection = sqlite3.connect(self.path, timeout=60.0)
            try:
                connection.execute("PRAGMA busy_timeout=60000")
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS variants (
                      variant_key TEXT PRIMARY KEY,
                      variant_id TEXT NOT NULL,
                      chrom TEXT NOT NULL,
                      pos INTEGER NOT NULL,
                      ref TEXT NOT NULL,
                      alt TEXT NOT NULL,
                      completed INTEGER NOT NULL,
                      elapsed_seconds REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS predictions (
                      variant_key TEXT NOT NULL,
                      gene_id TEXT NOT NULL,
                      tissue TEXT NOT NULL,
                      score REAL NOT NULL,
                      n_tracks INTEGER NOT NULL,
                      PRIMARY KEY (variant_key, gene_id, tissue)
                    );
                    CREATE INDEX IF NOT EXISTS predictions_tissue
                      ON predictions(tissue);
                    CREATE TABLE IF NOT EXISTS metadata (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL
                    );
                    """
                )
                self.connection = connection
                break
            except sqlite3.OperationalError as error:
                connection.close()
                if "locked" not in str(error).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PredictionStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def ensure_identity(self, identity: str) -> None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'model_identity'"
        ).fetchone()
        if row is not None and row[0] != identity:
            count = self.connection.execute(
                "SELECT COUNT(*) FROM variants WHERE completed = 1"
            ).fetchone()[0]
            if count:
                raise RuntimeError(
                    f"Prediction cache {self.path} belongs to another model/config; "
                    "use a different output directory"
                )
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('model_identity', ?)",
                (identity,),
            )

    def completed(self, key_or_record: str | VariantRecord) -> bool:
        key = (
            variant_key(key_or_record)
            if isinstance(key_or_record, VariantRecord)
            else key_or_record
        )
        row = self.connection.execute(
            "SELECT completed FROM variants WHERE variant_key = ?", (key,)
        ).fetchone()
        return bool(row and row[0])

    def save(
        self,
        record: VariantRecord,
        table: pd.DataFrame,
        elapsed_seconds: float = 0.0,
    ) -> None:
        key = variant_key(record)
        table = table.copy()
        if "variant_key" not in table:
            table["variant_key"] = key
        if not table.empty and set(table.variant_key) != {key}:
            raise ValueError("A save call may contain predictions for only one variant")
        table = normalize_prediction_table(table)
        with self.connection:
            self.connection.execute(
                "DELETE FROM predictions WHERE variant_key = ?", (key,)
            )
            self.connection.executemany(
                """INSERT INTO predictions
                   (variant_key, gene_id, tissue, score, n_tracks)
                   VALUES (?, ?, ?, ?, ?)""",
                table.itertuples(index=False, name=None),
            )
            self.connection.execute(
                """INSERT OR REPLACE INTO variants
                   (variant_key, variant_id, chrom, pos, ref, alt, completed,
                    elapsed_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                (
                    key,
                    record.variant_id,
                    record.chrom,
                    record.pos,
                    record.ref,
                    record.alt,
                    float(elapsed_seconds),
                ),
            )

    def clear(self, records: tuple[VariantRecord, ...] | list[VariantRecord]) -> None:
        with self.connection:
            for record in records:
                key = variant_key(record)
                self.connection.execute(
                    "DELETE FROM predictions WHERE variant_key = ?", (key,)
                )
                self.connection.execute(
                    "DELETE FROM variants WHERE variant_key = ?", (key,)
                )

    def read(self, tissue: str | None = None) -> pd.DataFrame:
        query = "SELECT variant_key, gene_id, tissue, score, n_tracks FROM predictions"
        params: tuple[str, ...] = ()
        if tissue is not None:
            query += " WHERE tissue = ?"
            params = (tissue,)
        return pd.read_sql_query(query, self.connection, params=params)

    def export(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.read().to_csv(path, sep="\t", index=False, compression="infer")
        return path
