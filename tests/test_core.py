import numpy as np
import pandas as pd
import pytest
import shapely

from locai.geo import LocalProjection, decay
from locai.geocode import geocode
from locai.metrics import CATALOG, DIMENSIONS
from locai.network import build_walk_network
from locai.profiles import group_masks, list_profiles, load_groups, load_profile


def test_decay_is_one_at_zero_and_decreasing():
    w = decay(np.array([0.0, 4.0, 8.0]), 4.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(np.exp(-1))
    assert w[2] < w[1]


@pytest.mark.parametrize("pid", list_profiles())
def test_profiles_are_consistent(pid):
    p = load_profile(pid)
    groups = load_groups()
    assert set(p.weights) <= set(DIMENSIONS)
    assert sum(p.weights.values()) == 100
    for dim, mets in p.metrics.items():
        for m in mets:
            assert CATALOG[m].dim == dim, f"{m} listed under {dim}"
    for g in [*p.competitors, *p.synergy]:
        assert g in groups, f"unknown group {g}"
    assert p.competition_curve["type"] in {"monotonic", "optimum"}


def test_group_masks_match_hierarchy_tokens_and_skip_closed():
    places = pd.DataFrame({
        "hierarchy": ["food_and_drink|cafe", "food_and_drink|restaurant|italian_restaurant", "food_and_drink|cafe", ""],
        "status": ["open", "open", "permanently_closed", "open"],
    })
    masks = group_masks(places, {"cafe": {"cafe"}, "restaurant": {"restaurant"}})
    assert masks["cafe"].tolist() == [True, False, False, False]
    assert masks["restaurant"].tolist() == [False, True, False, False]


def test_walk_network_splits_segments_at_connectors():
    proj = LocalProjection(45.0)
    kx = proj.kx
    # 200 m straight road along x with a connector in the middle; plus a motorway that must be excluded.
    line = shapely.LineString([(4.0, 45.0), (4.0 + 200 / kx, 45.0)])
    seg = pd.DataFrame({
        "subtype": ["road", "road"],
        "class": ["residential", "motorway"],
        "geometry": [shapely.to_wkb(line), shapely.to_wkb(line)],
        "connectors": [[("a", 0.0), ("b", 0.5), ("c", 1.0)], [("a", 0.0), ("c", 1.0)]],
    })
    net = build_walk_network(seg, proj)
    assert len(net.node_x) == 3
    assert sorted(net.edges_len.round()) == [100.0, 100.0]
    # A point 10 m off the road, 45 m from its start (on a 15 m sample): reached along its edge,
    # not via the nearest node.
    px, py = 4.0 * kx + 45, 45.0 * proj.ky + 10
    d = net.distances_from_point(px, py, 1000)
    assert sorted(d.round()) == [55.0, 65.0, 165.0]
    assert net.edges_near(px, py, 12).tolist() == [0]


ADDR = pd.DataFrame({
    "street": ["Rue de la République", "Rue de la République", "Rue de la République", "Place Bellecour"],
    "number": ["10", "12", "10", "3"],
    "postcode": ["69001", "69002", "69310", "69002"],
    "locality": ["Lyon 1er Arrondissement", "Lyon 2e Arrondissement", "Oullins-Pierre-Bénite", "Lyon 2e Arrondissement"],
    "lon": [4.8358, 4.8357, 4.8237, 4.8313],
    "lat": [45.7655, 45.7651, 45.7032, 45.7588],
})


def test_geocode_prefers_default_city_and_honours_hints():
    assert geocode("10 rue de la republique", ADDR, "Lyon").label.endswith("Lyon 1er Arrondissement")
    assert "Oullins" in geocode("10 rue de la République, Oullins", ADDR, "Lyon").label
    r = geocode("10 rue de la République, 69002", ADDR, "Lyon")
    assert not r.exact and r.label.startswith("≈ 12")


def test_geocode_coordinates_and_errors():
    r = geocode("45.7578, 4.8320", ADDR)
    assert (r.lat, r.lon) == (45.7578, 4.8320)
    with pytest.raises(ValueError):
        geocode("5 avenue nulle part", ADDR)
    with pytest.raises(ValueError):
        geocode("1 rue nowhere", ADDR)
    typo = geocode("10 rue de la Republiqe", ADDR, "Lyon")
    assert not typo.exact and typo.label.startswith("≈ 10 Rue de la République")
