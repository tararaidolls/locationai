"""Overture Maps (places, road segments, addresses) read straight from the public S3 bucket.

Uses row-group bbox statistics for pushdown, so only the needed slices of the global
GeoParquet files are downloaded.
"""
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs
import shapely

from ..config import OVERTURE_BUCKET, OVERTURE_REGION, OVERTURE_RELEASE


def _dataset(theme: str, type_: str) -> ds.Dataset:
    s3 = fs.S3FileSystem(anonymous=True, region=OVERTURE_REGION)
    path = f"{OVERTURE_BUCKET}/release/{OVERTURE_RELEASE}/theme={theme}/type={type_}/"
    return ds.dataset(path, filesystem=s3, format="parquet")


def _bbox_filter(bbox):
    x0, y0, x1, y1 = bbox
    return (
        (pc.field("bbox", "xmax") >= x0) & (pc.field("bbox", "xmin") <= x1)
        & (pc.field("bbox", "ymax") >= y0) & (pc.field("bbox", "ymin") <= y1)
    )


def fetch_places(bbox) -> pd.DataFrame:
    cols = ["id", "names", "taxonomy", "basic_category", "operating_status", "confidence", "addresses", "geometry"]
    df = _dataset("places", "place").to_table(columns=cols, filter=_bbox_filter(bbox)).to_pandas()
    pts = shapely.from_wkb(df["geometry"].values)
    return pd.DataFrame({
        "id": df["id"],
        "name": [n.get("primary") if n else None for n in df["names"]],
        "category": [t.get("primary") if t else None for t in df["taxonomy"]],
        "hierarchy": ["|".join(t["hierarchy"]) if t and t.get("hierarchy") is not None else "" for t in df["taxonomy"]],
        "basic_category": df["basic_category"],
        "status": df["operating_status"].fillna("open"),
        "confidence": df["confidence"],
        "address": [dict(a[0]) if a is not None and len(a) else None for a in df["addresses"]],
        "lon": shapely.get_x(pts),
        "lat": shapely.get_y(pts),
    })


def fetch_segments(bbox) -> pd.DataFrame:
    cols = ["id", "subtype", "class", "connectors", "geometry"]
    df = _dataset("transportation", "segment").to_table(columns=cols, filter=_bbox_filter(bbox)).to_pandas()
    df["connectors"] = [[(c["connector_id"], float(c["at"])) for c in cs] if cs is not None else [] for cs in df["connectors"]]
    return df


def fetch_addresses(bbox) -> pd.DataFrame:
    cols = ["street", "number", "postcode", "address_levels", "geometry"]
    df = _dataset("addresses", "address").to_table(columns=cols, filter=_bbox_filter(bbox)).to_pandas()
    pts = shapely.from_wkb(df["geometry"].values)
    return pd.DataFrame({
        "street": df["street"],
        "number": df["number"],
        "postcode": df["postcode"],
        "locality": [lv[0]["value"] if lv is not None and len(lv) else None for lv in df["address_levels"]],
        "lon": shapely.get_x(pts),
        "lat": shapely.get_y(pts),
    })


# Building subtypes that are not homes; NULL subtype is kept (most buildings are unlabelled).
NON_RESIDENTIAL = {"industrial", "commercial", "education", "religious", "outbuilding", "medical", "civic",
                   "agricultural", "service", "transportation", "military", "entertainment"}


def fetch_buildings(bbox, proj) -> pd.DataFrame:
    """Probable residential buildings: centroid, footprint area (m²) and floors when known."""
    cols = ["subtype", "height", "num_floors", "geometry"]
    df = _dataset("buildings", "building").to_table(columns=cols, filter=_bbox_filter(bbox)).to_pandas()
    df = df[~df["subtype"].isin(NON_RESIDENTIAL)]
    geoms = shapely.from_wkb(df["geometry"].values)
    cent = shapely.centroid(geoms)
    lon, lat = shapely.get_x(cent), shapely.get_y(cent)
    # Area in m² in the local projection (shoelace on projected rings is exact enough at city scale).
    projected = shapely.transform(geoms, lambda c: np.column_stack(proj.to_xy(c[:, 0], c[:, 1])))
    floors = df["num_floors"].astype(float).to_numpy()
    floors = np.where(np.isnan(floors), df["height"].astype(float).to_numpy() / 3.0, floors)
    return pd.DataFrame({"lon": lon, "lat": lat, "area_m2": shapely.area(projected), "floors": floors})


def fetch_communes(bbox) -> list[tuple[str, object]]:
    """Municipality / commune polygons (Overture division_area, admin level 2)."""
    cols = ["names", "subtype", "class", "geometry"]
    df = _dataset("divisions", "division_area").to_table(columns=cols, filter=_bbox_filter(bbox)).to_pandas()
    df = df[(df["subtype"] == "county") & (df["class"] == "land")]
    return [(n["primary"], g) for n, g in zip(df["names"], shapely.from_wkb(df["geometry"].values))]
