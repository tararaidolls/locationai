"""Offline geocoder over the city's address table (Overture/BAN for France, place addresses for Romania).

Accepts "lat, lon", "<number> <street>[, hint]" (French order) or "<street> [nr.] <number>[, hint]"
(Romanian order). Street types may be omitted or abbreviated ("Memorandumului 28", "Str. Horea 3").
Production should add an online geocoder per country (BAN, Nominatim, …) with this as a fallback.
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

_LATLON = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$")
_ADDR = re.compile(r"^\s*(\d+)\s*(?:(?:bis|ter|[a-z])\b)?\s*,?\s*([^,]+?)\s*(?:,(.*))?$", re.IGNORECASE)
_ADDR_STREET_FIRST = re.compile(
    r"^\s*([^,]*?[^\W\d_][^,]*?)\s*,?\s*(?:nr\.?\s*)?(\d{1,4}[a-z]?)\s*(?:,(.*))?$", re.IGNORECASE)
# Street-type words dropped for a second, looser match ("rue Paul Bert" ~ "Paul Bert", "Str. Horea" ~ "Horea").
_TYPE_WORDS = re.compile(
    r"^(rue|avenue|av|boulevard|bd|bld|place|pl|cours|quai|chemin|allee|impasse|route|"
    r"strada|str|bulevardul|b dul|bdul|calea|piata|aleea|al|splaiul|soseaua|sos|intrarea)\b\s*")
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


def _key(street_norm: str) -> str:
    return _TYPE_WORDS.sub("", street_norm).strip()


def _label(r) -> str:
    label = getattr(r, "label", None)
    if isinstance(label, str) and label:
        return label
    return f"{r.number} {r.street}, {r.postcode} {r.locality}"


def _parse(query: str):
    """(number, street, hint) from either address order; None if not an address."""
    m = _ADDR.match(query)
    if m:
        return m.group(1), m.group(2), m.group(3) or ""
    m = _ADDR_STREET_FIRST.match(query)
    if m:
        return m.group(2).lower(), m.group(1), m.group(3) or ""
    return None


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

    parsed = _parse(query)
    if not parsed:
        raise ValueError("Вкажіть адресу як «10 rue de la République», «Strada Memorandumului 28» "
                         "або координати «45.7640, 4.8357».")
    number, street_raw, hint = parsed
    street = _norm(street_raw)

    full: dict[str, list[str]] = {}
    loose: dict[str, list[str]] = {}
    for s_ in addresses["street"].dropna().unique():
        n = _norm(s_)
        full.setdefault(n, []).append(s_)
        loose.setdefault(_key(n), []).append(s_)
    match = difflib.get_close_matches(street, full.keys(), n=1, cutoff=0.85)
    if match:
        names, same_street = full[match[0]], _key(match[0]) == _key(street)
    else:
        match = difflib.get_close_matches(_key(street), loose.keys(), n=1, cutoff=0.85)
        if not match:
            raise ValueError(f"Вулицю «{street_raw}» не знайдено в адресному реєстрі міста.")
        names, same_street = loose[match[0]], match[0] == _key(street)
    rows = _pick_locality(addresses[addresses["street"].isin(names)], hint, default_locality)

    exact = rows[rows["number"].astype(str).str.lower() == number]
    if len(exact):
        r = exact.iloc[0]
        prefix = "" if same_street else "≈ "
        return GeocodeResult(float(r.lat), float(r.lon), prefix + _label(r), same_street)
    # Nearest house number on the same street.
    nums = pd.to_numeric(rows["number"].str.extract(r"(\d+)")[0], errors="coerce")
    r = rows.iloc[int(np.nanargmin(np.abs(nums.to_numpy() - int(re.match(r"\d+", number).group()))))]
    return GeocodeResult(float(r.lat), float(r.lon), "≈ " + _label(r), False)


def reverse(lat: float, lon: float, addresses: pd.DataFrame) -> str:
    d = (addresses["lat"] - lat) ** 2 + ((addresses["lon"] - lon) * np.cos(np.radians(lat))) ** 2
    r = addresses.loc[d.idxmin()]
    if isinstance(r.get("label"), str):
        return r["label"].split(",")[0]
    return f"{r.number} {r.street}"
