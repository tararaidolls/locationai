"""Offline geocoder over the city's address table (Overture addresses = BAN for France).

Accepts "lat, lon" or "<number> <street>[, anything]". Production should add an online
geocoder per country (BAN, CartoCiudad, PRG, Nominatim) with this as a fallback.
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

_LATLON = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$")
_ADDR = re.compile(r"^\s*(\d+)\s*(?:(?:bis|ter|[a-z])\b)?\s*,?\s*([^,]+?)\s*(?:,(.*))?$", re.IGNORECASE)
_POSTCODE = re.compile(r"\b(\d{5})\b")


@dataclass
class GeocodeResult:
    lat: float
    lon: float
    label: str
    exact: bool


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _pick_locality(rows: pd.DataFrame, hint: str, default_locality: str) -> pd.DataFrame:
    """Disambiguate same-named streets in different communes: postcode, then locality, then default city."""
    pc = _POSTCODE.search(hint)
    if pc and (rows["postcode"] == pc.group(1)).any():
        return rows[rows["postcode"] == pc.group(1)]
    loc_norm = rows["locality"].fillna("").map(_norm)
    h = _norm(hint)
    if h:
        hit = loc_norm.map(lambda loc: bool(loc) and (loc.startswith(h) or h.startswith(loc.split(" ")[0])))
        if hit.any():
            return rows[hit]
    hit = loc_norm.str.startswith(_norm(default_locality))
    return rows[hit] if hit.any() else rows


def geocode(query: str, addresses: pd.DataFrame, default_locality: str = "") -> GeocodeResult:
    m = _LATLON.match(query)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return GeocodeResult(lat, lon, f"{lat:.5f}, {lon:.5f}", True)

    m = _ADDR.match(query)
    if not m:
        raise ValueError("Вкажіть адресу як «10 rue de la République» або координати «45.7640, 4.8357».")
    number, street, hint = m.group(1), _norm(m.group(2)), m.group(3) or ""

    streets = addresses["street"].dropna().unique()
    norm_map: dict[str, list[str]] = {}
    for s in streets:
        norm_map.setdefault(_norm(s), []).append(s)
    match = difflib.get_close_matches(street, norm_map.keys(), n=1, cutoff=0.85)
    if not match:
        raise ValueError(f"Вулицю «{m.group(2)}» не знайдено в адресному реєстрі міста.")
    rows = _pick_locality(addresses[addresses["street"].isin(norm_map[match[0]])], hint, default_locality)

    same_street = match[0] == street
    exact = rows[rows["number"].astype(str).str.lower() == number]
    if len(exact):
        r = exact.iloc[0]
        prefix = "" if same_street else "≈ "
        return GeocodeResult(r.lat, r.lon, f"{prefix}{r.number} {r.street}, {r.postcode} {r.locality}", same_street)
    # Nearest house number on the same street.
    nums = pd.to_numeric(rows["number"].str.extract(r"(\d+)")[0], errors="coerce")
    r = rows.iloc[int(np.nanargmin(np.abs(nums.to_numpy() - int(number))))]
    return GeocodeResult(r.lat, r.lon, f"≈ {r.number} {r.street}, {r.postcode} {r.locality}", False)


def reverse(lat: float, lon: float, addresses: pd.DataFrame) -> str:
    d = (addresses["lat"] - lat) ** 2 + ((addresses["lon"] - lon) * np.cos(np.radians(lat))) ** 2
    r = addresses.loc[d.idxmin()]
    return f"{r.number} {r.street}"
