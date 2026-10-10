"""Pedestrian network built from Overture road segments.

Segments are split at their connectors; connectors become graph nodes and the pieces
between them become edges. Points (addresses, places, cell centroids) are snapped to the
nearest point on an edge, so a mid-block address is reached through both ends of its edge.
"""
from dataclasses import dataclass

import igraph as ig
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .config import NON_WALKABLE_CLASSES
from .geo import LocalProjection

SAMPLE_STEP_M = 15.0  # spacing of points sampled along edges for snapping


@dataclass
class EdgeSnap:
    """Snap of points to edges: the two end nodes and the walking distance to each (incl. the perpendicular)."""
    edge: np.ndarray
    u: np.ndarray
    v: np.ndarray
    du: np.ndarray
    dv: np.ndarray


@dataclass
class WalkNetwork:
    node_x: np.ndarray
    node_y: np.ndarray
    edges_u: np.ndarray
    edges_v: np.ndarray
    edges_len: np.ndarray
    # Points sampled along each edge's true geometry: coordinates, edge index, offset from edges_u.
    sample_x: np.ndarray
    sample_y: np.ndarray
    sample_edge: np.ndarray
    sample_offset: np.ndarray

    def __post_init__(self):
        n = len(self.node_x)
        m = coo_matrix((self.edges_len, (self.edges_u, self.edges_v)), shape=(n, n))
        self.graph: csr_matrix = (m + m.T).tocsr()
        self.sample_tree = cKDTree(np.column_stack([self.sample_x, self.sample_y]))

    def snap(self, x, y) -> EdgeSnap:
        perp, i = self.sample_tree.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
        e = self.sample_edge[i]
        off = self.sample_offset[i]
        return EdgeSnap(e, self.edges_u[e], self.edges_v[e], perp + off, perp + self.edges_len[e] - off)

    def distances_from_point(self, x: float, y: float, limit_m: float) -> np.ndarray:
        """Walking distance (m) from a point to every node; inf beyond limit."""
        s = self.snap(x, y)
        d = dijkstra(self.graph, directed=False, indices=[int(s.u[0]), int(s.v[0])], limit=limit_m)
        return np.minimum(d[0] + s.du[0], d[1] + s.dv[0])

    def edges_near(self, x: float, y: float, radius_m: float) -> np.ndarray:
        idx = self.sample_tree.query_ball_point([x, y], radius_m)
        if not idx:
            return self.snap(x, y).edge
        return np.unique(self.sample_edge[idx])

    def edge_betweenness(self, cutoff_m: float) -> np.ndarray:
        """Length-weighted edge betweenness limited to paths shorter than cutoff (space-syntax style)."""
        g = ig.Graph(n=len(self.node_x), edges=list(zip(self.edges_u.tolist(), self.edges_v.tolist())), directed=False)
        return np.asarray(g.edge_betweenness(weights=self.edges_len.tolist(), cutoff=cutoff_m, directed=False))

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})


def path_distances(node_dist: np.ndarray, snap: EdgeSnap) -> np.ndarray:
    """Walking distance to snapped targets given distances to all nodes."""
    return np.minimum(node_dist[snap.u] + snap.du, node_dist[snap.v] + snap.dv)


def build_walk_network(segments: pd.DataFrame, proj: LocalProjection) -> WalkNetwork:
    roads = segments[(segments["subtype"] == "road") & ~segments["class"].isin(NON_WALKABLE_CLASSES)]
    node_ids: dict[str, int] = {}
    node_xy: list[tuple[float, float]] = []
    eu, ev, el = [], [], []
    sx, sy, se, so = [], [], [], []

    geoms = shapely.from_wkb(roads["geometry"].values)
    for geom, conns in zip(geoms, roads["connectors"]):
        if geom is None or len(conns) < 2:
            continue
        coords = shapely.get_coordinates(geom)
        x, y = proj.to_xy(coords[:, 0], coords[:, 1])
        line = shapely.LineString(np.column_stack([x, y]))
        total = line.length
        conns = sorted(conns, key=lambda c: c[1])
        idx = []
        for cid, at in conns:
            if cid not in node_ids:
                p = line.interpolate(at, normalized=True)
                node_ids[cid] = len(node_xy)
                node_xy.append((p.x, p.y))
            idx.append(node_ids[cid])
        ats = [at for _, at in conns]
        for k in range(len(idx) - 1):
            if idx[k] == idx[k + 1]:
                continue
            start, length = ats[k] * total, max((ats[k + 1] - ats[k]) * total, 0.1)
            edge = len(eu)
            eu.append(idx[k])
            ev.append(idx[k + 1])
            el.append(length)
            offsets = np.append(np.arange(0.0, length, SAMPLE_STEP_M), length)
            pts = shapely.line_interpolate_point(line, start + offsets)
            sx.extend(shapely.get_x(pts))
            sy.extend(shapely.get_y(pts))
            se.extend([edge] * len(offsets))
            so.extend(offsets)

    xy = np.asarray(node_xy)
    return WalkNetwork(
        xy[:, 0], xy[:, 1], np.asarray(eu), np.asarray(ev), np.asarray(el, dtype=float),
        np.asarray(sx), np.asarray(sy), np.asarray(se), np.asarray(so, dtype=float),
    )
