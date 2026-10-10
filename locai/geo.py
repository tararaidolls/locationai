"""Small geometry helpers: a local metric projection good enough at city scale."""
import numpy as np


class LocalProjection:
    """Equirectangular projection around a reference latitude (error < 0.5% within ~20 km)."""

    def __init__(self, lat0: float):
        self.kx = 111_320.0 * np.cos(np.radians(lat0))
        self.ky = 110_540.0

    def to_xy(self, lon, lat):
        return np.asarray(lon) * self.kx, np.asarray(lat) * self.ky


def decay(t_min, tau_min):
    """Distance-decay weight w(t) = exp(-t / tau)."""
    return np.exp(-np.asarray(t_min) / tau_min)
