"""Meta HRSL population rasters (~30 m), aggregated to H3 cells."""
import h3
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from ..config import H3_RES, HRSL_BASE, HRSL_LAYERS

_GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt"}


def _read_layer(layer: str, bbox):
    url = f"/vsicurl/{HRSL_BASE}/{layer}/{layer}-latest.vrt"
    with rasterio.Env(**_GDAL_ENV), rasterio.open(url) as src:
        win = from_bounds(*bbox, transform=src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=0)
        transform = src.window_transform(win)
    return np.nan_to_num(arr, nan=0.0), transform


def population_by_h3(bbox) -> pd.DataFrame:
    """Return one row per H3 cell with population columns from every HRSL layer."""
    frames = []
    cells = None
    for col, layer in HRSL_LAYERS.items():
        arr, tr = _read_layer(layer, bbox)
        if cells is None:
            rows, cols_ = np.indices(arr.shape)
            xs = tr.c + (cols_ + 0.5) * tr.a
            ys = tr.f + (rows + 0.5) * tr.e
            cells = np.array([h3.latlng_to_cell(y, x, H3_RES) for y, x in zip(ys.ravel(), xs.ravel())])
            shape = arr.shape
        if arr.shape != shape:  # layers share the HRSL grid, but guard anyway
            raise ValueError(f"HRSL layer {layer} has unexpected shape {arr.shape} != {shape}")
        frames.append(pd.Series(arr.ravel(), index=cells, name=col).groupby(level=0).sum())
    out = pd.concat(frames, axis=1).fillna(0.0)
    out.index.name = "h3"
    return out[out["pop"] > 0]


def calibrate_to_census(cells: pd.DataFrame, buildings: pd.DataFrame, communes: list,
                        targets: dict[str, int], hrsl_weight: float = 0.5) -> tuple[pd.DataFrame, list[str]]:
    """Rescale population to census totals per commune and redistribute it partly by housing.

    Within each commune with a census target, a cell's share is
        hrsl_weight · (its share of HRSL population) + (1 − hrsl_weight) · (its share of residential floor area).
    HRSL keeps the density of tall blocks; floor area adds new neighbourhoods that HRSL predates.
    Demographic layers keep each cell's HRSL composition (commune average for cells new to HRSL).
    """
    import shapely

    layers = list(HRSL_LAYERS)
    b = buildings.assign(h3=[h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(buildings["lat"], buildings["lon"])])
    floor_area = (b["area_m2"] * b["floors"].fillna(1.0).clip(1, 30)).groupby(b["h3"]).sum()
    out = cells.set_index("h3")[layers].reindex(cells["h3"].tolist() + [c for c in floor_area.index if c not in set(cells["h3"])])
    out = out.fillna(0.0)
    latlng = np.array([h3.cell_to_latlng(c) for c in out.index])
    notes = []
    for name, poly in communes:
        if name not in targets:
            continue
        inside = shapely.contains_xy(poly, latlng[:, 1], latlng[:, 0])
        if not inside.any():
            continue
        idx = out.index[inside]
        hrsl = out.loc[idx, "pop"]
        area = floor_area.reindex(idx).fillna(0.0)
        share = hrsl_weight * hrsl / max(hrsl.sum(), 1e-9) + (1 - hrsl_weight) * area / max(area.sum(), 1e-9)
        new_pop = targets[name] * share
        mix = out.loc[idx, layers].div(hrsl.replace(0, np.nan), axis=0)
        avg_mix = out.loc[idx, layers].sum() / max(hrsl.sum(), 1e-9)
        mix = mix.fillna(avg_mix)
        out.loc[idx, layers] = mix.mul(new_pop, axis=0).to_numpy()
        notes.append(f"{name}: HRSL {hrsl.sum():,.0f} → перепис {targets[name]:,.0f}")
    out = out[out["pop"] > 0].reset_index().rename(columns={"index": "h3"})
    return out, notes
