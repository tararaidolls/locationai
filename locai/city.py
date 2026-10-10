"""City data bundle: everything precomputed once per city and cached on disk.

  places.parquet     Overture places (+ projected x/y and snap to the walk network)
  network.npz        pedestrian graph
  betweenness.npy    edge betweenness (street centrality)
  cells.parquet      H3 cells with HRSL population layers (+ snap to the walk network)
  transit_rail.npy   tram/subway lines densified into points every RAIL_STEP_M
  addresses.parquet  Overture/BAN addresses for offline geocoding
"""
import logging
import time
from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from .config import City, city_dir
from .geo import LocalProjection
from .network import WalkNetwork, build_walk_network
from .sources import overture, population

log = logging.getLogger(__name__)

BETWEENNESS_CUTOFF_M = 1500.0
RAIL_STEP_M = 25.0
TRANSIT_RAIL_CLASSES = {"tram", "subway", "light_rail", "funicular"}


@dataclass
class CityData:
    city: City
    proj: LocalProjection
    places: pd.DataFrame
    cells: pd.DataFrame
    net: WalkNetwork
    betweenness: np.ndarray
    rail_tree: cKDTree | None
    addresses: pd.DataFrame

    def __post_init__(self):
        self.places_tree = cKDTree(self.places[["x", "y"]].to_numpy())
        self.cells_tree = cKDTree(self.cells[["x", "y"]].to_numpy())


def _attach_xy_and_node(df: pd.DataFrame, proj: LocalProjection, net: WalkNetwork) -> pd.DataFrame:
    df = df.copy()
    df["x"], df["y"] = proj.to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    s = net.snap(df["x"].to_numpy(), df["y"].to_numpy())
    df["snap_u"], df["snap_v"], df["snap_du"], df["snap_dv"] = s.u, s.v, s.du, s.dv
    return df


def build_city(city: City) -> None:
    d = city_dir(city)
    proj = LocalProjection((city.bbox[1] + city.bbox[3]) / 2)
    t = time.time()

    log.info("Fetching Overture road segments…")
    segments = overture.fetch_segments(city.data_bbox)
    net = build_walk_network(segments, proj)
    net.save(d / "network.npz")
    log.info("Walk network: %d nodes, %d edges (%.0fs)", len(net.node_x), len(net.edges_u), time.time() - t)

    rail = segments[(segments["subtype"] == "rail") & segments["class"].isin(TRANSIT_RAIL_CLASSES)]
    pts = []
    for geom in shapely.from_wkb(rail["geometry"].values):
        c = shapely.get_coordinates(geom)
        x, y = proj.to_xy(c[:, 0], c[:, 1])
        line = shapely.LineString(np.column_stack([x, y]))
        for s in np.arange(0, line.length, RAIL_STEP_M):
            p = line.interpolate(s)
            pts.append((p.x, p.y))
    np.save(d / "transit_rail.npy", np.asarray(pts).reshape(-1, 2))

    log.info("Computing street centrality (betweenness, cutoff %.0f m)…", BETWEENNESS_CUTOFF_M)
    np.save(d / "betweenness.npy", net.edge_betweenness(BETWEENNESS_CUTOFF_M))
    log.info("Centrality done (%.0fs)", time.time() - t)

    log.info("Fetching Overture places…")
    places = overture.fetch_places(city.data_bbox)
    _attach_xy_and_node(places, proj, net).to_parquet(d / "places.parquet")

    log.info("Fetching HRSL population…")
    pop = population.population_by_h3(city.data_bbox).reset_index()
    latlng = np.array([h3.cell_to_latlng(c) for c in pop["h3"]])
    pop["lat"], pop["lon"] = latlng[:, 0], latlng[:, 1]
    _attach_xy_and_node(pop, proj, net).to_parquet(d / "cells.parquet")

    log.info("Fetching addresses for the offline geocoder…")
    overture.fetch_addresses(city.data_bbox).to_parquet(d / "addresses.parquet")
    log.info("City %s built in %.0fs → %s", city.key, time.time() - t, d)


def load_city(city: City) -> CityData:
    d = city_dir(city)
    if not (d / "addresses.parquet").exists():
        raise FileNotFoundError(f"City '{city.key}' is not built yet. Run: python -m locai build-city {city.key}")
    rail = np.load(d / "transit_rail.npy")
    return CityData(
        city=city,
        proj=LocalProjection((city.bbox[1] + city.bbox[3]) / 2),
        places=pd.read_parquet(d / "places.parquet"),
        cells=pd.read_parquet(d / "cells.parquet"),
        net=WalkNetwork.load(d / "network.npz"),
        betweenness=np.load(d / "betweenness.npy"),
        rail_tree=cKDTree(rail) if len(rail) else None,
        addresses=pd.read_parquet(d / "addresses.parquet"),
    )
