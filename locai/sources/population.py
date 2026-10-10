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
