"""Static configuration: cities, data releases, model constants."""
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

OVERTURE_RELEASE = "2026-09-23.1"
OVERTURE_BUCKET = "overturemaps-us-west-2"
OVERTURE_REGION = "us-west-2"

HRSL_BASE = "https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs"
# Meta High Resolution Settlement Layer: population per ~30 m pixel, by demographic group.
HRSL_LAYERS = {
    "pop": "hrsl_general",
    "women": "hrsl_women",
    "youth_15_24": "hrsl_youth_15_24",
    "elderly_60_plus": "hrsl_elderly_60_plus",
    "children_under_5": "hrsl_children_under_five",
}

H3_RES = 9
WALK_SPEED_M_PER_MIN = 75.0  # 4.5 km/h
# Road classes excluded from the pedestrian graph.
NON_WALKABLE_CLASSES = {"motorway"}


@dataclass(frozen=True)
class City:
    key: str
    name: str
    country: str
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (lon/lat)
    default_locality: str  # preferred commune when a street name exists in several
    buffer_deg: float = 0.025  # data is fetched this far beyond bbox to avoid edge effects (~2–3 km)
    # Census populations per commune (Overture division names). When set, HRSL is rescaled to these
    # totals and partly redistributed by residential floor area (see population.calibrate_to_census).
    population_targets: dict[str, int] | None = None

    @property
    def data_bbox(self) -> tuple[float, float, float, float]:
        b = self.buffer_deg
        return (self.bbox[0] - b, self.bbox[1] - b, self.bbox[2] + b, self.bbox[3] + b)

    def contains(self, lon: float, lat: float) -> bool:
        return self.bbox[0] <= lon <= self.bbox[2] and self.bbox[1] <= lat <= self.bbox[3]


CITIES = {
    "lyon": City("lyon", "Lyon + Villeurbanne", "FR", (4.77, 45.70, 4.92, 45.81), "Lyon"),
    # Cluj-Napoca with Florești (the large commuter town to the west).
    "cluj": City("cluj", "Cluj-Napoca + Florești", "RO", (23.47, 46.72, 23.70, 46.81), "Cluj-Napoca",
                 # INS, Recensământul populației 2021. Florești figure to be double-checked against INS.
                 population_targets={"Cluj-Napoca": 286_598, "Florești": 52_735}),
}


def city_dir(city: City) -> Path:
    d = DATA_DIR / city.key
    d.mkdir(parents=True, exist_ok=True)
    return d


def built_cities() -> list[str]:
    """Cities whose data has been prepared with `build-city`."""
    return [k for k, c in CITIES.items() if (DATA_DIR / c.key / "addresses.parquet").exists()]
