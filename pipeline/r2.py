from pathlib import Path

import numpy as np
from thucia.core.cases import r2
from thucia.core.cases import read_nc
from thucia.viz import plot_all_admin2

dirstem = Path("data") / "cases" / "PER"
models = ["baseline", "climate", "sarima", "tcn"]
plot_regions = True

for model in models:
    filename = str(dirstem / f"{model}_cases_quantiles.nc")
    df = read_nc(filename)

    r2_model = r2(
        df,
        "prediction",
        "Cases",
        transform=np.log1p,
        df_filter={"quantile": 0.50},
    )
    print(f"{model} (R^2): {r2_model:.3f}")
    if plot_regions:
        plot_all_admin2(
            df,
            transform=np.log1p,
        )
