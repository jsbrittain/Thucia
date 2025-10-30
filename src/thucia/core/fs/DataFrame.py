import logging
import re
from contextlib import AbstractContextManager
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class DBConnect(AbstractContextManager[duckdb.DuckDBPyConnection]):
    """
    Context manager for DuckDB connections.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = ":memory:" if db_path is None else db_path
        self._persistent_con: duckdb.DuckDBPyConnection | None = None

        if self.db_path == ":memory:":
            # Persistent connection for in-memory DBs
            self._persistent_con = duckdb.connect(self.db_path)
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    # `with dbconnect as con:` (read-only)
    def __enter__(self) -> duckdb.DuckDBPyConnection:
        if self.db_path == ":memory:":
            return self._persistent_con
        self._temp_con = duckdb.connect(self.db_path, read_only=True)
        return self._temp_con

    def __exit__(self, exc_type, exc_val, exc_tb):
        temp = getattr(self, "_temp_con", None)
        if temp is not None:
            try:
                temp.close()
            except Exception:
                pass
            delattr(self, "_temp_con")

    # context manager used by __call__()
    @contextmanager
    def connection(self, *args, **kwargs):
        if self.db_path == ":memory:":
            yield self._persistent_con
            return

        con = duckdb.connect(self.db_path, *args, **kwargs)
        try:
            yield con
        finally:
            try:
                con.close()
            except Exception:
                pass

    # `with dbconnect(read_only=False) as con:`
    def __call__(self, *args, **kwargs):
        return self.connection(*args, **kwargs)

    def __del__(self):
        con = getattr(self, "_persistent_con", None)
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
            self._persistent_con = None


class DataFrame:
    """
    Lightweight wrapper around a DuckDB table.
    Access columns lazily as pandas Series/DataFrames.
    """

    def __init__(
        self,
        db_path: str | None = None,
        db_file: str | Path | None = None,
        new_file: bool = False,
        table: str | None = None,
        df: pd.DataFrame | None = None,
    ):
        # db_path is the old name for db_file; support both for now
        if db_file and not db_path:
            db_path = db_file

        # Read parameters from existing database
        if db_path and (table is None) and not new_file:
            table = self.get_table(db_path)

        # Defaults
        db_path = (
            ":memory:" if db_path is None else db_path
        )  # in-memory database, can replace with temporary file
        table = "data" if table is None else table

        # Delete existing file if new_file is requested
        if new_file and db_path != ":memory:":
            Path(db_path).unlink(missing_ok=True)

        # Assign attributes
        self.db_path = db_path
        self.table = table
        self.dbconnect = DBConnect(self.db_path)

        # Initialise with pandas dataframe
        if df is not None:
            self.write_df(df)

    def __len__(self) -> int:
        """Return number of rows in the table."""
        query = f"SELECT COUNT(*) FROM {self.table}"
        try:
            with self.dbconnect as con:
                return con.execute(query).fetchone()[0]
        except duckdb.CatalogException:
            return 0

    def query_df(self, query: str) -> pd.DataFrame:
        """Run a query and return a pandas DataFrame, restoring Period dtypes"""
        with self.dbconnect as con:
            df = con.execute(query).fetch_df()
            # try to read column metadata for this table and restore Period columns
            try:
                q = (
                    "SELECT column_name, is_period, freq, how FROM __column_metadata__ "
                    "WHERE table_name = ?"
                )
                rows = con.execute(q, [self.table]).fetchall()
            except Exception:
                rows = []

        if rows:
            for col_name, is_period, freq, how in rows:
                if is_period and col_name in df.columns and freq:
                    # ensure timestamp-like then convert back to Period
                    df[col_name] = pd.to_datetime(df[col_name])
                    df[col_name] = df[col_name].dt.to_period(freq)

        return df

    def __getitem__(self, key: Any) -> pd.DataFrame | pd.Series:
        """Return a column or subset as pandas object."""
        # handle a single column name
        if isinstance(key, str):
            query = f"SELECT {key} FROM {self.table}"
            return self.query_df(query)[key]
        # handle list/tuple of columns
        elif isinstance(key, (list, tuple)):
            cols = ", ".join(key)
            query = f"SELECT {cols} FROM {self.table}"
            return self.query_df(query)
        elif isinstance(key, slice):
            # support slicing by row index
            start = key.start or 0
            stop = key.stop or self.__len__()
            query = f"SELECT * FROM {self.table} LIMIT {stop - start} OFFSET {start}"
            return self.query_df(query)
        elif isinstance(key, pd.Series):
            # support boolean mask series for filtering rows
            if key.dtype == bool:
                indices = key[key].index.tolist()
                if not indices:
                    return pd.DataFrame(columns=self.columns)
                indices_str = ", ".join(map(str, indices))
                query = f"SELECT * FROM {self.table} WHERE ROWID IN ({indices_str})"
                return self.query_df(query)
            else:
                raise TypeError("Series key must be of boolean dtype for row filtering")
        else:
            raise TypeError("Key must be a column name or list of column names")

    @property
    def df(self) -> pd.DataFrame:
        """Return the entire table as a pandas DataFrame and restore Period dtypes
        using stored metadata.
        """
        return self.query_df(f"SELECT * FROM {self.table}")

    def get_table(self, db_path: str):
        # Check file for existing tables
        if not Path(db_path).exists():
            return None
        try:
            with duckdb.connect(db_path) as con:
                tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

            # Exclude internal / metadata tables (names starting with '__')
            user_tables = [t for t in tables if not t.startswith("__")]

            if not user_tables:
                raise ValueError(
                    "No user tables found in the database. "
                    "Found only metadata/internal tables."
                )
            if len(user_tables) > 1:
                raise ValueError(
                    "Multiple user tables found; please specify which table to "
                    "use via the 'table' argument."
                )
            # Exactly one user table — select it
            table = user_tables[0]
            return table
        except ValueError:
            return None
        return None

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.query_df(f"SELECT * FROM {self.table} LIMIT {n}").fetch_df()

    def query(self, sql_filter: str) -> pd.DataFrame:
        """Run a WHERE-style filter and return DataFrame."""
        q = f"SELECT * FROM {self.table} WHERE {sql_filter}"
        return self.query_df(q)

    @property
    def columns(self) -> list[str]:
        with self.dbconnect as con:
            return [row[0] for row in con.execute(f"DESCRIBE {self.table}").fetchall()]

    def __repr__(self):
        return f"<DataFrame table='{self.table}' db='{self.db_path}'>"

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Work on a shallow copy so we don't mutate caller's DataFram
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
        return df_to_write, meta

    def _create_table_with_enum(self, df, con=None):
        ctx = self.dbconnect(read_only=False) if con is None else nullcontext(con)
        with ctx as con:
            table_exists = (
                con.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables "
                    f"WHERE table_name = '{self.table}'"
                ).fetchone()[0]
                > 0
            )

            # Create table
            if not table_exists:
                # no data insert
                con.execute(f"CREATE TABLE {self.table} AS SELECT * FROM df LIMIT 0")

                # Ensure GID_2 is stored as ENUM with its full category list
                cats = df["GID_2"].cat.categories.tolist()
                cats_escaped = [c.replace("'", "''") for c in cats]
                enum_list = ",".join(f"'{c}'" for c in cats_escaped)
                con.execute(
                    f"ALTER TABLE {self.table} ALTER GID_2 SET DATA TYPE ENUM({enum_list})"
                )

            # Insert data
            con.register("df", df)
            con.execute(f"INSERT INTO {self.table} SELECT * FROM df")
            con.unregister("df")

    def write_df(self, df: pd.DataFrame, con=None):
        """Write a pandas DataFrame to the DuckDB table, replacing existing data.
        Minimal metadata support: detects Period dtypes and stores column_name
         -> freq in __column_metadata__.
        """
        df_to_write, meta = self._prepare_df(df)

        # Write DataFrame into DuckDB (only open once for read-write atomicity)
        ctx = self.dbconnect(read_only=False) if con is None else nullcontext(con)
        with ctx as con:
            con.execute(f"DROP TABLE IF EXISTS {self.table}")
            self._create_table_with_enum(df_to_write, con=con)

            # Ensure metadata table exists, then replace rows for this table
            con.execute(
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
            con.execute(
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
                con.executemany(
                    "INSERT INTO __column_metadata__ VALUES (?, ?, ?, ?, ?, ?)", rows
                )

    def append(self, df: pd.DataFrame):
        """Append a pandas DataFrame to the existing DuckDB table, updating metadata
        as needed.
        """

        # We keep the read-write connection open for the duration of this method
        # to avoid multiple open/close cycles which would introduce non-atomicity.
        with self.dbconnect(read_only=False) as con:
            table_exists = (
                con.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables "
                    f"WHERE table_name = '{self.table}'"
                ).fetchone()[0]
                > 0
            )

            if not table_exists:
                # First write, use write_df to ensure metadata is created
                self.write_df(df, con=con)
            else:
                # Subsequently append
                df_to_write, meta = self._prepare_df(df)
                self._create_table_with_enum(df_to_write, con=con)
