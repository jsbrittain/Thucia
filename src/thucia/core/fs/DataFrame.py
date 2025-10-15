import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class DataFrame:
    """
    Lightweight wrapper around a DuckDB table.
    Access columns lazily as pandas Series/DataFrames.
    """

    def __init__(
        self,
        db_path: str | None = None,
        table: str | None = None,
        df: pd.DataFrame | None = None,
    ):
        # Read parameters from existing database
        if db_path and (table is None):
            # Check file for existing tables
            if Path(db_path).exists():
                with duckdb.connect(db_path) as con:
                    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

                    # Exclude internal / metadata tables (names starting with '__')
                    user_tables = [t for t in tables if not t.startswith("__")]

                    if not user_tables:
                        raise ValueError(
                            "No user tables found in the database. Found only metadata/internal tables."
                        )
                    if len(user_tables) > 1:
                        raise ValueError(
                            "Multiple user tables found; please specify which table to use via the 'table' argument."
                        )
                    # Exactly one user table — select it
                    table = user_tables[0]

        # Defaults
        db_path = (
            ":memory:" if db_path is None else db_path
        )  # in-memory database, can replace with temporary file
        table = "data" if table is None else table

        # Assign attributes
        self.db_path = db_path
        self.table = table
        self.con = duckdb.connect(self.db_path)

        # Initialise with pandas dataframe
        if df is not None:
            self.write_df(df)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con:
            self.con.close()

    def __len__(self) -> int:
        """Return number of rows in the table."""
        query = f"SELECT COUNT(*) FROM {self.table}"
        return self.con.execute(query).fetchone()[0]

    def __getitem__(self, key: Any) -> pd.DataFrame | pd.Series:
        """Return a column or subset as pandas object."""
        # handle a single column name
        if isinstance(key, str):
            query = f"SELECT {key} FROM {self.table}"
            return self.con.execute(query).fetch_df()[key]
        # handle list/tuple of columns
        elif isinstance(key, (list, tuple)):
            cols = ", ".join(key)
            query = f"SELECT {cols} FROM {self.table}"
            return self.con.execute(query).fetch_df()
        else:
            raise TypeError("Key must be a column name or list of column names")

    @property
    def df(self) -> pd.DataFrame:
        """Return the entire table as a pandas DataFrame and restore Period dtypes using stored metadata."""
        df = self.con.execute(f"SELECT * FROM {self.table}").fetch_df()

        # try to read column metadata for this table and restore Period columns
        try:
            q = "SELECT column_name, is_period, freq, how FROM __column_metadata__ WHERE table_name = ?"
            rows = self.con.execute(q, [self.table]).fetchall()
        except Exception:
            rows = []

        if rows:
            for col_name, is_period, freq, how in rows:
                if is_period and col_name in df.columns and freq:
                    # ensure timestamp-like then convert back to Period
                    df[col_name] = pd.to_datetime(df[col_name])
                    df[col_name] = df[col_name].dt.to_period(freq)

        return df

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.con.execute(f"SELECT * FROM {self.table} LIMIT {n}").fetch_df()

    def query(self, sql_filter: str) -> pd.DataFrame:
        """Run a WHERE-style filter and return DataFrame."""
        q = f"SELECT * FROM {self.table} WHERE {sql_filter}"
        return self.con.execute(q).fetch_df()

    @property
    def columns(self) -> list[str]:
        return [row[0] for row in self.con.execute(f"DESCRIBE {self.table}").fetchall()]

    def __repr__(self):
        return f"<DataFrame table='{self.table}' db='{self.db_path}'>"

    def write_df(self, df: pd.DataFrame):
        """Write a pandas DataFrame to the DuckDB table, replacing existing data.
        Minimal metadata support: detects Period dtypes and stores column_name -> freq in __column_metadata__.
        """
        # Work on a shallow copy so we don't mutate caller's DataFrame
        df_to_write = df.copy()

        # Build metadata for period columns (store freq and edge used)
        meta = {}
        for col in df_to_write.columns:
            if isinstance(df_to_write[col].dtype, pd.PeriodDtype):
                freq = re.search(r"period\[(.+)\]", df_to_write[col].dtype.name).group(
                    1
                )
                meta[col] = {"is_period": True, "freq": freq, "how": "end"}

        # Convert Period columns to timestamp (end of period) for storage
        for col, info in meta.items():
            df_to_write[col] = df_to_write[col].dt.to_timestamp(
                how=info.get("how", "end")
            )

        # Convert some common columns to categorical to save space (keep this minimal)
        categorical_cols = [
            "GID_1",
            "GID_2",
            "Status",
        ]
        for col in categorical_cols:
            if col in df_to_write.columns:
                try:
                    df_to_write[col] = df_to_write[col].astype("category")
                except Exception:
                    logging.debug(
                        f"Could not convert column {col} to category; leaving as-is."
                    )

        # Write DataFrame into DuckDB (register + create/insert pattern)
        # Register the DataFrame so DuckDB can select from it
        self.con.register("df", df_to_write)
        self.con.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.con.execute(f"CREATE TABLE {self.table} AS SELECT * FROM df")
        self.con.unregister("df")

        # Ensure metadata table exists, then replace rows for this table
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS __column_metadata__ (
                table_name VARCHAR,
                column_name VARCHAR,
                is_period BOOLEAN,
                freq VARCHAR,
                how VARCHAR,
                updated_at TIMESTAMP
            )
            """
        )
        # Delete existing metadata for this table, then insert new rows
        self.con.execute(
            "DELETE FROM __column_metadata__ WHERE table_name = ?", [self.table]
        )

        rows = []
        for col, info in meta.items():
            rows.append(
                (
                    self.table,
                    col,
                    True,
                    info.get("freq"),
                    info.get("how", "end"),
                    pd.Timestamp.utcnow(),
                )
            )

        if rows:
            self.con.executemany(
                "INSERT INTO __column_metadata__ VALUES (?, ?, ?, ?, ?, ?)", rows
            )
