"""Address tables for countries without an open address register in Overture.

For Romania there is no Overture address theme, so the table is assembled from the free-form
addresses of businesses (Overture places): "Strada Memorandumului 28", "Str. Horea, nr. 3",
"Cluj, Calea Florești nr. 58". Coverage is partial (only where a business is listed), which is
why the geocoder also falls back to the nearest house number on the street.
"""
import re
import unicodedata

import pandas as pd

# Abbreviation → canonical street type (Romanian).
RO_TYPES = {
    "str": "Strada", "strada": "Strada",
    "bd": "Bulevardul", "bdul": "Bulevardul", "b-dul": "Bulevardul", "bulevardul": "Bulevardul", "blvd": "Bulevardul",
    "calea": "Calea", "cal": "Calea",
    "piata": "Piața", "piața": "Piața", "p-ta": "Piața",
    "aleea": "Aleea", "al": "Aleea",
    "splaiul": "Splaiul", "spl": "Splaiul",
    "soseaua": "Șoseaua", "șoseaua": "Șoseaua", "sos": "Șoseaua", "șos": "Șoseaua",
    "intrarea": "Intrarea", "intr": "Intrarea",
}
# Leading comma-separated parts that are place names, not streets.
RO_PLACE_PARTS = re.compile(
    r"^(cluj(-napoca)?|cluj napoca|municipiul cluj-napoca|jud(e[tț]ul)?\.? cluj|rom[aâ]nia|loc\.?\s*\S+|"
    r"flore[sș]ti|baciu|apahida|feleacu|chinteni|gil[aă]u)$",
    re.IGNORECASE,
)
ROAD_CODE = re.compile(r"^(E|DN|DJ|DC|A)\s?\d", re.IGNORECASE)
# House numbers: up to 3 digits plus a letter; "58-60" keeps the first. Four digits are years
# in street names ("Bulevardul 21 Decembrie 1989"), not house numbers.
_NUM = r"(\d{1,3}[a-z]?)(?:\s*-\s*\d{1,3}[a-z]?)?"
_NR = re.compile(r"\b(?:nr|num[aă]r(?:ul)?)\.?\s*" + _NUM + r"\b", re.IGNORECASE)
_TRAILING_NUM = re.compile(r"\s" + _NUM + r"\s*$", re.IGNORECASE)
_LOC_PREFIX = re.compile(r"^(loc|localitatea|sat|com)\.?\s+", re.IGNORECASE)


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def parse_ro_freeform(text: str | None) -> tuple[str, str] | None:
    """Split a Romanian free-form address into (street, number); None if it has no usable street + number."""
    if not text or not text.strip():
        return None
    parts = []
    for p in (p.strip() for p in text.split(",")):
        if not p:
            continue
        if parts and _NR.match(p):  # "Strada X, nr. 5" → one part
            parts[-1] = f"{parts[-1]} {p}"
        else:
            parts.append(p)
    while parts and RO_PLACE_PARTS.match(parts[0]):
        parts = parts[1:]
    if not parts or ROAD_CODE.match(parts[0]):
        return None
    for part in parts:
        if ROAD_CODE.match(part):
            continue
        m = _NR.search(part)
        if m:
            street, number = part[:m.start()], m.group(1)
        else:
            m = _TRAILING_NUM.search(part)
            if not m:
                continue
            street, number = part[:m.start()], m.group(1)
        street = re.sub(r"^([^\W\d_]+)\.(?=\S)", r"\1. ", street.strip())  # "Str.Horea" → "Str. Horea"
        street = _LOC_PREFIX.sub("", re.sub(r"\s+", " ", street).strip(" .,-"))
        if len(re.sub(r"[^A-Za-zÀ-ž]", "", street)) >= 3:
            break
    else:
        return None
    number = number.lower()
    words = street.split(" ")
    canon = RO_TYPES.get(_ascii(words[0]).rstrip("."))
    if canon:
        words = [canon] + words[1:]
    street = " ".join(w[:1].upper() + w[1:] if i else w for i, w in enumerate(words))
    return street, number


def ro_locality(locality: str | None, postcode: str | None, default: str) -> str:
    pc = (postcode or "").strip()
    if pc.startswith("400"):
        return default
    loc = (locality or "").strip()
    if not loc or "metropolitan" in _ascii(loc) or "romania" in _ascii(loc):
        return default if not pc else loc or default
    return loc.split(",")[0].strip()


def addresses_from_places(places: pd.DataFrame, default_locality: str) -> pd.DataFrame:
    """Build an address table (street, number, postcode, locality, label, lon, lat) from place addresses."""
    rows = []
    for a, lon, lat in zip(places["address"], places["lon"], places["lat"]):
        if not a:
            continue
        parsed = parse_ro_freeform(a.get("freeform"))
        if not parsed:
            continue
        street, number = parsed
        locality = ro_locality(a.get("locality"), a.get("postcode"), default_locality)
        rows.append((street, number, (a.get("postcode") or "").strip() or None, locality, lon, lat))
    df = pd.DataFrame(rows, columns=["street", "number", "postcode", "locality", "lon", "lat"])
    if df.empty:
        return df.assign(label=pd.Series(dtype=str))
    # Spelling variants of one place ("Floresti", "Florești"; "Cluj Napoca") → the most common spelling.
    akey = df["locality"].map(lambda v: re.sub(r"[^a-z]", "", _ascii(v)))
    canon = df.groupby(akey)["locality"].agg(lambda v: v.value_counts().index[0])
    df["locality"] = akey.map(canon)
    if re.sub(r"[^a-z]", "", _ascii(default_locality)) in canon.index:
        df.loc[akey == re.sub(r"[^a-z]", "", _ascii(default_locality)), "locality"] = default_locality
    # Several businesses share one address: keep one point per (street, number, locality).
    key = df["street"].map(_ascii) + "|" + df["number"] + "|" + df["locality"].map(_ascii)
    df = (df.assign(_k=key).groupby("_k", as_index=False)
          .agg(street=("street", "first"), number=("number", "first"), postcode=("postcode", "first"),
               locality=("locality", "first"), lon=("lon", "median"), lat=("lat", "median")))
    df["label"] = df["street"] + " " + df["number"] + ", " + df["locality"]
    return df
