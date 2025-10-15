import random
from pathlib import Path

import numpy as np
import pandas as pd
from thucia.core.cases import write_db

n = 250  # Number of cases to generate

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

path = Path(__file__).parent / "test_data"
path.mkdir(exist_ok=True)

# Source dates
dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="ME").to_list()
gid2s = [f"GID_{i:02d}" for i in range(1, 11)]

# Generate line list dataset: Date, GID_2, Cases
df = pd.DataFrame(
    {
        "Date": np.random.choice(dates, size=n, replace=True),
        "GID_1": ["GID_000"] * n,
        "GID_2": np.random.choice(gid2s, size=n, replace=True),
    }
)
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")  # Format dates as strings

df.to_csv(path / "cases.csv", index=False)
write_db(df, path / "cases")
