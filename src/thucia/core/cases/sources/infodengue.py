# Infodengue case-source driver.
#
# Infodengue (https://info.dengue.mat.br) is a Brazilian arboviral surveillance
# service covering dengue / chikungunya / zika by municipality (IBGE code). The
# driver is intentionally disease-generic: the disease, ISO3 country, states,
# and date window are all parameters passed to fetch() -- "dengue" is only a
# default value, never a hard-coded constant, and the class name refers to the
# provider, not the disease.
from __future__ import annotations

from io import BytesIO
from io import StringIO
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pandas as pd
import requests

from . import CaseSource
from . import case_registry

#: IBGE municipality codebook (zip), used to resolve municipality names/codes.
#: See https://www.ibge.gov.br/explica/codigos-dos-municipios.php
_IBGE_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/estrutura_territorial/"
    "divisao_territorial/2024/DTB_2024.zip"
)
_IBGE_FILE = "DTB_2024/RELATORIO_DTB_BRASIL_2024_MUNICIPIOS.xls"

#: Infodengue alert API endpoint.
_API_URL = "https://info.dengue.mat.br/api/alertcity"


@case_registry.register()
class InfodengueSource(CaseSource):
    ref = "infodengue"
    name = "Infodengue"

    def __init__(self, **params) -> None:
        #: Default parameters, overridable (per-call) in fetch().
        self.params: dict = params

    # ------------------------------------------------------------------ #
    # Reusable pieces (network / filesystem), overridable for testing.   #
    # ------------------------------------------------------------------ #
    def _resolve_geocodes(
        self,
        iso3: str = "BRA",
        states: Optional[list[str]] = None,
        municipalities_path: str | Path = _IBGE_FILE,
        **params,
    ) -> pd.DataFrame:
        """Municipality codebook filtered to `states` (``geocode/state/municipality``)."""
        if iso3.lower() != "bra":
            raise ValueError(f"Infodengue only covers Brazil municipalities; iso3={iso3!r}")
        if not Path(municipalities_path).exists():
            self._download_municipality_codes(municipalities_path)
        df = pd.read_excel(
            municipalities_path,
            header=6,
            sheet_name="DTB_Municípios",
            usecols=[
                "Nome_UF",  # State name
                "Código Município Completo",  # Municipality code
                "Nome_Município",  # Municipality name
            ],
        )
        df.rename(
            columns={
                "Código Município Completo": "geocode",
                "Nome_Município": "municipality",
                "Nome_UF": "state",
            },
            inplace=True,
        )
        if states:
            df = df[df["state"].isin(states)]
        return df

    def _download_municipality_codes(self, municipalities_path: str | Path) -> None:
        url = self.params.get("ibge_url", _IBGE_URL)
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise OSError(f"Error downloading municipality codes: {response.status_code}")
        with ZipFile(BytesIO(response.content)) as zip_file:
            target = Path(municipalities_path).parent
            zip_file.extractall(target)

    def _query_state(
        self,
        geocode: str,
        disease: str = "dengue",
        ey_start: Optional[int] = None,
        ew_start: Optional[int] = None,
        ey_end: Optional[int] = None,
        ew_end: Optional[int] = None,
        **params,
    ) -> pd.DataFrame:
        """Monthly case totals for one municipality geocode from the alert API."""
        current_date = pd.Timestamp.now().isocalendar()
        ey_start = ey_start or current_date.year - 1
        ew_start = ew_start or max(1, current_date.week - 1)
        ey_end = ey_end or current_date.year
        ew_end = ew_end or current_date.week

        search_filter = {
            "geocode": geocode,
            "disease": disease,  # dengue | chikungunya | zika
            "format": "json",
            "ey_start": ey_start,
            "ew_start": ew_start,
            "ey_end": ey_end,
            "ew_end": ew_end,
        }
        url = self.params.get("api_url", _API_URL)
        full_url = url + "?" + "&".join(f"{k}={v}" for k, v in search_filter.items())

        response = None
        for _ in range(5):
            try:
                response = requests.get(full_url, timeout=30)
                break
            except requests.exceptions.RequestException:
                continue
        if response is None or response.status_code != 200:
            raise OSError(f"Error fetching data: {response.status_code}")
        payload = response.json()
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"API Error: {payload.get('error_message')}")

        df = pd.read_json(StringIO(response.text))
        if df.empty:
            return df

        df["week_start"] = pd.to_datetime(df["data_iniSE"], unit="ms")
        df["end_of_month"] = df["week_start"] + pd.offsets.MonthEnd(0)
        df = df[["casos", "end_of_month"]]
        df = df.groupby("end_of_month").sum().reset_index()
        df["geocode"] = geocode
        return df

    def _query_states(self, geocodes, disease: str = "dengue", **params) -> pd.DataFrame:
        frames = []
        for geocode in geocodes:
            sub = self._query_state(geocode, disease=disease, **params)
            if sub.empty:
                continue
            sub["geocode"] = geocode
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def attach_admin_info(self, df: pd.DataFrame, geocodes: pd.DataFrame) -> pd.DataFrame:
        """Attach IBGE ``state``/``municipality`` names to queried rows."""
        out = df.merge(geocodes[["geocode", "state", "municipality"]], on="geocode", how="left")
        if out.isnull().values.any():
            import warnings

            warnings.warn("Some geocodes could not be matched with municipality info", UserWarning)
        return out

    # ------------------------------------------------------------------ #
    # Public entry point.                                                #
    # ------------------------------------------------------------------ #
    def fetch(self, **params) -> pd.DataFrame:
        """Query Infodengue case data (disease-generic) and return a DataFrame.

        Parameters (any may be supplied; defaults come from the constructor):
          disease : str           one of "dengue" / "chikungunya" / "zika".
          iso3    : str           country (Infodengue coverage is "BRA").
          states  : list[str]|None IBGE state names to filter to (None = all).
          geocodes: list[str]|None explicit IBGE municipality codes (skips codebook).
          ey_start/ew_start/ey_end/ew_end: epidemiological year/week window.
          align   : bool          align region names to GADM admin regions.
          out_path: str|Path|None write the result to a NetCDF file.
        """
        p = {**self.params, **params}
        disease = p.get("disease", "dengue")
        iso3 = p.get("iso3", "BRA")
        states = p.get("states")
        geocodes = p.get("geocodes")
        align = p.get("align", True)
        out_path = p.get("out_path")

        if geocodes is None:
            geocodes_df = self._resolve_geocodes(iso3=iso3, states=states)
            geocode_list = geocodes_df["geocode"].unique()
        else:
            geocode_list = list(geocodes)
            geocodes_df = pd.DataFrame({"geocode": geocode_list})

        df = self._query_states(
            geocode_list, disease=disease, **{k: v for k, v in p.items() if k != "disease"}
        )
        if df.empty:
            return df
        df = self.attach_admin_info(df, geocodes_df)
        df.drop(columns=["geocode"], inplace=True)
        df.rename(
            columns={
                "state": "ADM1",
                "municipality": "ADM2",
                "end_of_month": "Date",
                "casos": "Cases",
            },
            inplace=True,
        )

        if align:
            from thucia.core.geo import align_admin2_regions

            df = align_admin2_regions(df, "ADM1", "ADM2", iso3=iso3)
        if out_path is not None:
            from thucia.core.fs import write_nc

            write_nc(df, out_path)
        return df


__all__ = ["InfodengueSource"]
