import os

import pandas as pd
import pytest
from thucia.core.cases.sources import CaseSource
from thucia.core.cases.sources import case_registry
from thucia.core.cases.sources import load_case_source
from thucia.core.cases.sources import load_sources
from thucia.core.cases.sources.infodengue import InfodengueSource
from thucia.core.fs import read_nc
from thucia.core.registry import PluginNotFoundError


def test_real_case_sources_self_register():
    load_sources()
    assert case_registry.has("infodengue")
    cls = case_registry.get("infodengue")
    assert issubclass(cls, CaseSource)
    assert cls.ref == "infodengue"


def test_load_case_source_returns_configured_instance():
    src = load_case_source("infodengue", iso3="BRA")
    assert isinstance(src, InfodengueSource)
    assert src.params["iso3"] == "BRA"


def test_load_case_source_unknown_ref_raises():
    with pytest.raises(PluginNotFoundError):
        load_case_source("not-a-source")


def test_base_fetch_not_implemented():
    with pytest.raises(NotImplementedError):
        CaseSource().fetch()


def test_load_sources_isolates_broken_module(tmp_path, monkeypatch):
    # A module that raises on import must not stop the other drivers loading.
    # Use unique package/module names so we don't collide with the geo plugin
    # loader test (which uses a generic "sources"/"fine").
    import sys

    stem = "casesrc_tmp_sources"
    pkg = tmp_path / stem
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken.py").write_text("raise RuntimeError('boom')")
    (pkg / "finesrc.py").write_text(
        "from thucia.core.cases.sources import case_registry, CaseSource\n"
        "@case_registry.register()\n"
        "class FineSrc(CaseSource):\n"
        "    ref = 'casesrc_fine'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in list(sys.modules):
        if mod.startswith(stem):
            sys.modules.pop(mod, None)
    try:
        plugins = load_sources(plugin_dir=str(pkg), module_stem=stem)
    finally:
        case_registry.unregister("casesrc_fine")
        for mod in list(sys.modules):
            if mod.startswith(stem):
                sys.modules.pop(mod, None)
    assert "broken" not in plugins
    assert "casesrc_fine" in plugins


def _make_source(monkeypatch):
    """An InfodengueSource with the network paths stubbed out (offline)."""
    src = load_case_source("infodengue", iso3="BRA")

    def fake_geocodes(iso3="BRA", states=None):
        return pd.DataFrame(
            {
                "geocode": ["1100205", "1100015"],
                "state": ["Rondônia", "Rondônia"],
                "municipality": ["Porto Velho", "Alta Floresta do Oeste"],
            }
        )

    calls = {"disease": None}

    def fake_query_state(geocode, disease="dengue", **kw):
        calls["disease"] = disease
        return pd.DataFrame(
            {
                "casos": [3, 5],
                "end_of_month": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            }
        )

    monkeypatch.setattr(src, "_resolve_geocodes", fake_geocodes)
    monkeypatch.setattr(src, "_query_state", fake_query_state)
    return src, calls


def test_fetch_uses_disease_as_parameter(monkeypatch):
    # The disease must be threaded through as a param, never hard-coded.
    src, calls = _make_source(monkeypatch)
    out = src.fetch(disease="zika", align=False)
    assert calls["disease"] == "zika"
    assert set(out.columns) == {"ADM1", "ADM2", "Cases", "Date"}
    assert out["Cases"].tolist() == [3, 5, 3, 5]
    assert out["Date"].nunique() == 2


def test_fetch_default_disease(monkeypatch):
    src, calls = _make_source(monkeypatch)
    src.fetch(align=False)
    assert calls["disease"] == "dengue"


def test_fetch_align_mocked(monkeypatch):
    src, calls = _make_source(monkeypatch)
    captured = {}

    def fake_align(df, *a, **k):
        captured.update(k)
        df = df.copy()
        df["GID_1"] = "BRA.1_1"
        df["GID_2"] = "BRA.1.1_1"
        return df

    monkeypatch.setattr("thucia.core.geo.align_admin2_regions", fake_align)
    out = src.fetch(disease="dengue", align=True, iso3="BRA")
    assert captured["iso3"] == "BRA"
    assert "GID_2" in out.columns


def test_fetch_writes_and_round_trips_nc(monkeypatch, tmp_path):
    src, _ = _make_source(monkeypatch)
    out_path = os.path.join(tmp_path, "cases.nc")
    src.fetch(align=False, out_path=out_path)
    assert os.path.exists(out_path)
    back = read_nc(out_path)
    assert set(["ADM1", "ADM2", "Cases", "Date"]).issubset(back.columns)
    assert len(back) == len(src.fetch(align=False))


def test_resolve_geocodes_rejects_non_bra(monkeypatch):
    src = load_case_source("infodengue", iso3="BRA")
    with pytest.raises(ValueError, match="Brazil"):
        src._resolve_geocodes(iso3="FRA")


def test_resolve_geocodes_reads_and_filters(monkeypatch, tmp_path):
    # Mock read_excel so we don't need a real IBGE codebook file; the path just
    # has to exist to avoid the download branch.
    fake_xls = tmp_path / "codes.xls"
    fake_xls.touch()
    monkeypatch.setattr(
        "thucia.core.cases.sources.infodengue.pd.read_excel",
        lambda *a, **k: pd.DataFrame(
            {
                "Código Município Completo": ["1100205", "1100015", "1200179"],
                "Nome_Município": ["Porto Velho", "Alta Floresta do Oeste", "Acre"],
                "Nome_UF": ["Rondônia", "Rondônia", "Acre"],
            }
        ),
    )
    src = load_case_source("infodengue", iso3="BRA")
    df = src._resolve_geocodes(states=["Rondônia"], municipalities_path=str(fake_xls))
    assert set(df.columns) == {"geocode", "state", "municipality"}
    assert df["geocode"].tolist() == ["1100205", "1100015"]


def test_constructor_params_are_fetch_defaults(monkeypatch):
    # Disease passed at construction becomes the fetch default.
    src, calls = _make_source(monkeypatch)
    src.params["disease"] = "chikungunya"
    src.fetch(align=False)
    assert calls["disease"] == "chikungunya"


# --------------------------------------------------------------------------- #
# Real network-path logic exercised offline via a stubbed `requests.get`.      #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status_code=200, text="[]", payload=None):
        import json

        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload) if payload is not None else text

    def json(self):
        import json

        return self._payload if self._payload is not None else json.loads(self.text)


def test_query_state_parses_and_aggregates_monthly(monkeypatch):
    # Raw weekly rows (data_iniSE in ms) collapse into monthly totals.
    payload = [
        {"data_iniSE": 1577836800000, "casos": 2},  # 2020-01-01
        {"data_iniSE": 1578009600000, "casos": 1},  # 2020-01-03 (same month)
        {"data_iniSE": 1580515200000, "casos": 3},  # 2020-02-01
    ]
    monkeypatch.setattr(
        "thucia.core.cases.sources.infodengue.requests.get",
        lambda *a, **k: _FakeResponse(text="", payload=payload),
    )
    src = load_case_source("infodengue", iso3="BRA")
    df = src._query_state(geocode="1100205", disease="zika")
    assert df["casos"].tolist() == [3, 3]
    assert df["geocode"].tolist() == ["1100205", "1100205"]


def test_query_state_http_error_raises(monkeypatch):
    monkeypatch.setattr(
        "thucia.core.cases.sources.infodengue.requests.get",
        lambda *a, **k: _FakeResponse(status_code=500),
    )
    src = load_case_source("infodengue", iso3="BRA")
    with pytest.raises(OSError):
        src._query_state(geocode="1100205")


def test_query_state_api_error_raises(monkeypatch):
    monkeypatch.setattr(
        "thucia.core.cases.sources.infodengue.requests.get",
        lambda *a, **k: _FakeResponse(
            text="", payload={"error": "x", "error_message": "bad request"}
        ),
    )
    src = load_case_source("infodengue", iso3="BRA")
    with pytest.raises(RuntimeError, match="bad request"):
        src._query_state(geocode="1100205")


def test_download_municipality_codes_writes_zip(monkeypatch, tmp_path):
    from io import BytesIO
    from zipfile import ZipFile

    buf = BytesIO()
    with ZipFile(buf, "w") as zf:
        zf.writestr("RELATORIO.xls", b"placeholder")

    class _DownloadResponse:
        status_code = 200
        content = buf.getvalue()

    monkeypatch.setattr(
        "thucia.core.cases.sources.infodengue.requests.get",
        lambda *a, **k: _DownloadResponse(),
    )
    out = tmp_path / "DTB_2024"
    src = load_case_source("infodengue", iso3="BRA")
    src._download_municipality_codes(str(out / "RELATORIO_DTB_BRASIL.xls"))
    assert (out / "RELATORIO.xls").exists()
