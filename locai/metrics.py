"""Raw metric computation for one point (methodology §3)."""
from dataclasses import dataclass

import numpy as np

from .city import RAIL_STEP_M, CityData
from .config import WALK_SPEED_M_PER_MIN
from .geo import decay
from .network import EdgeSnap, path_distances
from .profiles import Profile, group_masks, load_groups

HUFF_LAMBDA = 2.0
HUFF_DETOUR = 1.3        # straight-line → network distance factor
HUFF_MIN_DIST_M = 50.0
RETAIL_RADIUS_MIN = 3.0  # ~200 m walk
TRANSIT_RADIUS_MIN = 7.0
RAIL_RADIUS_M = 300.0
RAIL_STOP_SPACING_M = 400.0
CENTRALITY_RADIUS_M = 40.0


@dataclass(frozen=True)
class MetricDef:
    dim: str
    label: str
    unit: str
    direction: int  # +1: more is better, -1: less is better
    quality: str    # A / B / C (methodology §8)
    source: str


CATALOG: dict[str, MetricDef] = {
    "D1": MetricDef("D", "Населення в зоні (зважене)", "осіб", +1, "B", "Meta HRSL ~30 м"),
    "D2": MetricDef("D", "Цільовий сегмент (зважений)", "осіб", +1, "B", "Meta HRSL, демографічні шари"),
    "W1": MetricDef("W", "Офіси та бізнес-послуги поруч", "зваж. к-сть", +1, "C", "Overture Places"),
    "W2": MetricDef("W", "Навчальні заклади поруч", "зваж. к-сть", +1, "C", "Overture Places"),
    "W3": MetricDef("W", "Готелі поруч", "зваж. к-сть", +1, "C", "Overture Places"),
    "A1": MetricDef("A", "Громадський транспорт", "індекс", +1, "C", "Overture Places + лінії трамваю/метро"),
    "A2": MetricDef("A", "Центральність вулиці", "betweenness", +1, "B", "мережа Overture Transportation"),
    "A3": MetricDef("A", "Щільність торгівлі й послуг (~200 м)", "закладів", +1, "B", "Overture Places"),
    "C1": MetricDef("C", "Конкуренти в зоні (зважено)", "зваж. к-сть", -1, "B", "Overture Places"),
    "C2": MetricDef("C", "Насиченість: сегмент на 1 конкурента", "осіб", +1, "B", "похідна D2 / C1"),
    "C4": MetricDef("C", "Очікувані клієнти за моделлю Хаффа", "осіб", +1, "B", "похідна, модель Хаффа"),
    "S1": MetricDef("S", "Синергія з сусідніми закладами", "індекс", +1, "B", "Overture Places"),
}

DIMENSIONS = {
    "D": "Попит мешканців",
    "W": "Денне населення",
    "A": "Трафік і доступність",
    "C": "Конкуренція",
    "S": "Синергія",
    "R": "Ризик і стабільність",
}


def _snap_of(df) -> EdgeSnap:
    return EdgeSnap(None, df["snap_u"].to_numpy(), df["snap_v"].to_numpy(), df["snap_du"].to_numpy(), df["snap_dv"].to_numpy())


class MetricEngine:
    """Computes raw metrics for points in one city for one profile."""

    def __init__(self, city: CityData, profile: Profile):
        self.city = city
        self.profile = profile
        p, c = city.places, city.cells
        self.masks = group_masks(p, load_groups())
        self.place_snap = _snap_of(p)
        self.place_xy = p[["x", "y"]].to_numpy()
        self.cell_snap = _snap_of(c)
        self.cell_xy = c[["x", "y"]].to_numpy()
        self.cell_pop = c["pop"].to_numpy()
        self.cell_seg = np.clip(sum(coef * c[layer].to_numpy() for layer, coef in profile.segment.items()), 0, None)
        comp = np.zeros(len(p), dtype=bool)
        for g in profile.competitors:
            comp |= self.masks[g]
        self.comp_mask = comp
        self.comp_xy = self.place_xy[comp]

    def times(self, x: float, y: float, max_min: float):
        """Walking minutes from (x, y) to every place and every population cell."""
        dist = self.city.net.distances_from_point(x, y, max_min * WALK_SPEED_M_PER_MIN)
        t_places = path_distances(dist, self.place_snap) / WALK_SPEED_M_PER_MIN
        t_cells = path_distances(dist, self.cell_snap) / WALK_SPEED_M_PER_MIN
        return t_places, t_cells

    def compute(self, x: float, y: float) -> dict[str, float]:
        pr = self.profile
        T, tau = pr.walk_minutes, pr.tau_min
        t_places, t_cells = self.times(x, y, max(T, TRANSIT_RADIUS_MIN))

        in_cells = t_cells <= T
        w_cells = np.where(in_cells, decay(t_cells, tau), 0.0)
        in_places = t_places <= T
        w_places = np.where(in_places, decay(t_places, tau), 0.0)

        def wcount(group: str) -> float:
            return float(w_places[self.masks[group]].sum())

        m: dict[str, float] = {}
        m["D1"] = float((self.cell_pop * w_cells).sum())
        m["D2"] = float((self.cell_seg * w_cells).sum())
        m["W1"] = wcount("offices")
        m["W2"] = wcount("education")
        m["W3"] = wcount("lodging")

        w_transit = np.where(t_places <= TRANSIT_RADIUS_MIN, decay(t_places, tau), 0.0)
        rail_m = 0.0
        if self.city.rail_tree is not None:
            rail_m = len(self.city.rail_tree.query_ball_point([x, y], RAIL_RADIUS_M)) * RAIL_STEP_M
        m["A1"] = float(w_transit[self.masks["transit"]].sum()) + rail_m / RAIL_STOP_SPACING_M

        m["A2"] = float(self.city.betweenness[self.city.net.edges_near(x, y, CENTRALITY_RADIUS_M)].max())
        m["A3"] = float(((t_places <= RETAIL_RADIUS_MIN) & self.masks["retail_activity"]).sum())

        m["C1"] = float(w_places[self.comp_mask].sum())
        m["C2"] = m["D2"] / (m["C1"] + 1.0)
        m["C4"] = self._huff(x, y, in_cells)
        m["S1"] = float(sum(beta * wcount(g) for g, beta in pr.synergy.items()))
        return m

    def _huff(self, x: float, y: float, in_cells: np.ndarray) -> float:
        """Expected customers of a new average store (Huff model, equal attractiveness)."""
        cxy = self.cell_xy[in_cells]
        if len(cxy) == 0:
            return 0.0
        seg = self.cell_seg[in_cells]
        reach = self.profile.walk_minutes * WALK_SPEED_M_PER_MIN * 1.5
        near = self.city.places_tree.query_ball_point([x, y], reach)
        near = [i for i in near if self.comp_mask[i]]
        d_new = np.maximum(np.hypot(cxy[:, 0] - x, cxy[:, 1] - y) * HUFF_DETOUR, HUFF_MIN_DIST_M)
        u_new = d_new ** -HUFF_LAMBDA
        if near:
            pxy = self.place_xy[near]
            d = np.maximum(np.hypot(cxy[:, None, 0] - pxy[None, :, 0], cxy[:, None, 1] - pxy[None, :, 1]) * HUFF_DETOUR, HUFF_MIN_DIST_M)
            u_comp = (d ** -HUFF_LAMBDA).sum(axis=1)
        else:
            u_comp = 0.0
        return float((seg * u_new / (u_new + u_comp)).sum())
