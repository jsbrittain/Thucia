import geopandas as gpd

from pathlib import Path
from thucia.core.fs import cache_folder
from rasterstats import zonal_stats


def raster_stats_gid2(tif_file, gid_2s: list[str], stats=["mean"]):
    """
    Calculate zonal statistics for a given GeoDataFrame of polygons against a raster
    file.

    Parameters:

    Returns:
    gpd.GeoDataFrame: The input GeoDataFrame with additional columns for the
    calculated statistics.
    """

    iso3 = set(map(lambda x: x.split(".")[0], gid_2s))
    if len(iso3) != 1:
        raise ValueError("All filters must be for the same ISO3 country code.")
    iso3 = iso3.pop()

    file_path = Path(cache_folder) / "geo" / iso3 / f"gadm41_{iso3}.gpkg"
    if not file_path.exists():
        raise FileNotFoundError(f"GeoPackage file for {iso3} not found at {file_path}")

    polygons = gpd.read_file(str(file_path), layer="ADM_ADM_2")
    polygons = polygons[polygons["GID_2"].isin(gid_2s)]
    stats = zonal_stats(polygons, tif_file, stats=stats)
    for measure in stats[0].keys():
        polygons[measure] = [stat[measure] for stat in stats]
    # Drop geometry
    polygons = polygons.drop(columns=["geometry"])
    return polygons
