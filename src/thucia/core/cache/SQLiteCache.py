import sqlite3
from itertools import islice
from typing import List

import pandas as pd

from .CacheBase import CacheBase


class SQLiteCache(CacheBase):
    def __init__(
        self,
        cache_file: str,
        columntypes: dict,  # {'column_name': 'type'}
        keys: List[str],
        tablename: str = "stats",
    ):
        self.cache_file = cache_file
        self.columntypes = columntypes
        self.columns = list(columntypes.keys())
        self.tablename = tablename
        self.keys = keys
        self.key_names = ", ".join(self.keys)

        if not all([key in self.columns for key in self.keys]):
            raise ValueError(f"Keys {self.keys} must be in columns {self.columns}")

    def get_record(self, metric, date, GID_2):
        if not self.cache_file.exists():
            return None

        with sqlite3.connect(self.cache_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM {self.tablename}
                WHERE metric = ? AND Date = ? AND GID_2 = ?
                """,
                (metric, date.strftime("%Y-%m-%d"), GID_2),
            )
            row = cursor.fetchone()

        if row:
            return pd.DataFrame([row], columns=self.columns)
        return None

    def _chunked_iterable(self, iterable, size):
        """Yield successive chunks from iterable."""
        it = iter(iterable)
        while chunk := list(islice(it, size)):
            yield chunk

    def get_records(self, d: dict[str, List[str | int | float]]) -> pd.DataFrame:
        if not self.cache_file.exists():
            return pd.DataFrame(columns=self.columns)

        # Dictionary of lists to list of tuples
        keynames = list(d.keys())
        keys = list(zip(*[d[key] for key in keynames]))

        return self._get_records_batched(
            db_path=self.cache_file,
            keynames=keynames,
            keys=keys,
        )

    def _get_records_batched(
        self,
        db_path: str,
        keynames: List[str],
        keys: List[tuple],
        batch_size: int = 1000,
    ):
        rows = []
        with sqlite3.connect(db_path) as conn:
            for batch in self._chunked_iterable(keys, batch_size):
                keynames_placeholder = ",".join(keynames)
                keys_placeholder = ",".join(["?"] * len(keynames))
                batch_placeholder = ",".join([f"({keys_placeholder})"] * len(batch))
                flat_values = [item for triple in batch for item in triple]

                sql = f"""
                SELECT * FROM {self.tablename}
                WHERE ({keynames_placeholder}) IN ({batch_placeholder})
                """

                rows.extend(conn.execute(sql, flat_values).fetchall())

        return (
            pd.DataFrame(rows, columns=self.columns)
            if rows
            else pd.DataFrame(columns=self.columns)
        )

    def add_records(self, records):
        if not self.cache_file.parent.exists():
            self.cache_file.parent.mkdir(parents=True)

        with sqlite3.connect(self.cache_file) as conn:
            cursor = conn.cursor()
            column_definitions = ", ".join(
                f"{col} {self.columntypes[col]}" for col in self.columns
            )
            column_names = ", ".join(self.columns)
            placeholder_names = ", ".join(["?"] * len(self.columns))
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.tablename} (
                    {column_definitions},
                    PRIMARY KEY ({self.key_names})
                )
                """
            )
            for _, row in records.iterrows():
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.tablename} (
                        {column_names}
                    ) VALUES (
                        {placeholder_names}
                    )
                    """,
                    tuple(row[col] for col in self.columns),
                )
