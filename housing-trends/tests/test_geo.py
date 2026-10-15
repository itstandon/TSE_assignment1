"""Tests for the ZIP-centroid validation. These exist because the first model asked for DFW centroids returned a
FABRICATED answer that passed a bounding-box check (see AI_USE.md): latitude pinned at 32.95, longitude marching in
exact -0.03 steps in ZIP order, seven ZIPs on one identical point. Every check below is a defence that reply defeated.
"""
import pytest

from housing_trends.geo import (DFW_BBOX, PROBES, degeneracy_report, haversine_km, neighbour_order, numeric_order,
                                probe_errors, validate_centroids)

DOWNTOWN = (32.7815, -96.7954)


def fabricated(zips):
    """The failure mode seen in practice: coordinates interpolated from the ZIP number."""
    return {z: (32.95, -96.80 - 0.02 * i) for i, z in enumerate(sorted(zips))}


def test_haversine_matches_known_distances():
    assert haversine_km(DOWNTOWN, DOWNTOWN) == 0
    assert haversine_km(DOWNTOWN, PROBES["75287"]) == pytest.approx(25, abs=4)      # downtown -> far north Dallas
    assert haversine_km((32.0, -97.0), (33.0, -97.0)) == pytest.approx(111, abs=2)  # one degree of latitude


def test_validate_centroids_rejects_bad_shape_and_anything_outside_dfw():
    clean, errors = validate_centroids({"75201": [32.7815, -96.7954], "7520": [32.8, -96.8], "75202": "nope",
                                        "75203": [40.7, -74.0], "75204": [32.8]})
    assert set(clean) == {"75201"}
    assert len(errors) == 4 and any("not a 5-digit ZIP" in e for e in errors) and any("bounding box" in e for e in errors)


def test_validate_centroids_rejects_a_non_object():
    assert validate_centroids([1, 2])[1]


def test_probe_errors_catch_a_misplaced_known_zip():
    assert probe_errors({"75201": DOWNTOWN}) == []
    bad = probe_errors({"75201": (32.95, -96.20)})
    assert len(bad) == 1 and "75201" in bad[0] and "km" in bad[0]
    assert probe_errors({}) == []                       # nothing to check is not a failure


def test_degeneracy_report_catches_duplicate_points():
    out = degeneracy_report({"75201": DOWNTOWN, "75202": DOWNTOWN, "75230": PROBES["75230"]})
    assert any("share a centroid" in o for o in out)


def test_degeneracy_report_catches_an_arithmetic_coordinate_sequence():
    out = degeneracy_report(fabricated([f"752{i:02d}" for i in range(10, 30)]))
    assert any("constant step" in o or "same latitude" in o for o in out)


def test_degeneracy_report_accepts_the_real_probe_positions():
    assert degeneracy_report(dict(PROBES)) == []


def test_the_fabricated_reply_is_rejected_although_it_is_inside_the_bounding_box():
    # a snapshot-sized ZIP set, as the real reply was: the sequence checks need more than a handful of points
    cen = fabricated(sorted(set(PROBES) | {f"752{i:02d}" for i in range(10, 45)}))
    assert all(DFW_BBOX[0] <= la <= DFW_BBOX[1] and DFW_BBOX[2] <= lo <= DFW_BBOX[3] for la, lo in cen.values())
    assert validate_centroids({z: list(c) for z, c in cen.items()})[1] == []        # bbox check alone passes it
    assert degeneracy_report(cen) and probe_errors(cen)                             # the real checks catch it


# ---- the ordering itself
ZIPS = ["75201", "75202", "75203", "75204"]


def test_numeric_order_is_the_fallback_and_is_deterministic():
    o = numeric_order(ZIPS)
    assert o["75201"] == ["75202", "75203", "75204"] and "75201" not in o["75201"]
    assert neighbour_order(ZIPS, {}) == o and neighbour_order(ZIPS, None) == o


def test_centroids_reorder_the_neighbours_by_real_distance():
    cen = {"75201": (32.78, -96.79), "75202": (32.90, -96.79), "75203": (32.79, -96.79), "75204": (32.85, -96.79)}
    assert neighbour_order(ZIPS, cen)["75201"] == ["75203", "75204", "75202"]        # numeric order would be 02, 03, 04


def test_zips_without_a_centroid_keep_the_numeric_order_and_come_last():
    cen = {"75201": (32.78, -96.79), "75204": (32.79, -96.79)}
    assert neighbour_order(ZIPS, cen)["75201"] == ["75204", "75202", "75203"]
    assert neighbour_order(ZIPS, cen)["75202"] == ["75201", "75203", "75204"]        # no centroid -> numeric
