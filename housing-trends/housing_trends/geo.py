"""Geographic neighbour ordering for ZIP codes.

The Scanner's Pooling moves borrow listings from "adjacent" ZIPs. Without coordinates the only ordering available is
the NUMERICALLY nearest ZIP code, which is wrong often enough to matter: 75201 and 75202 are neighbours downtown, but
75208 and 75209 sit on opposite sides of the Trinity river.

Division of labour (the same rule as every other LLM node here: the model supplies knowledge, code does the arithmetic):
  * an LLM is asked ONCE, offline, for the approximate centroid (lat, lon) of each ZIP in the snapshot
    (scripts/build_zip_centroids.py). The reply is validated against a DFW bounding box and written to
    data/snapshot/zip_centroids.json, which is shipped with the repo.
  * at run time this module only reads that file and sorts by great-circle distance. No network, no model, no clock:
    the ordering is a frozen data artefact exactly like the listings snapshot.

A ZIP with no usable centroid keeps the numeric ordering, so a partial or missing file degrades instead of failing.
"""
from __future__ import annotations

import json
import math
import pathlib

# Bounding box of the Dallas-Fort Worth metro: anything outside it is not a DFW ZIP centroid and is rejected.
DFW_BBOX = (32.0, 33.6, -97.9, -96.2)          # (lat_min, lat_max, lon_min, lon_max)
CENTROIDS_PATH = pathlib.Path(__file__).resolve().parent.parent / "data/snapshot/zip_centroids.json"

# Probe ZIPs: well-known Dallas neighbourhoods whose approximate centroid is public knowledge. They are mixed into
# every batch of the build script WITHOUT being marked, and the reply is thrown away if it misplaces one. A model
# that cannot locate downtown Dallas is not a usable source for the ZIPs we cannot check.
PROBES = {
    "75201": (32.7815, -96.7954),      # Downtown
    "75208": (32.7490, -96.8290),      # North Oak Cliff / Bishop Arts
    "75214": (32.8240, -96.7540),      # Lakewood
    "75230": (32.8970, -96.7870),      # Preston Hollow
    "75243": (32.9120, -96.7340),      # Northeast Dallas
    "75287": (32.9990, -96.8380),      # Far North Dallas
}
PROBE_TOLERANCE_KM = 15.0              # generous: these are approximate centroids, not survey points
MIN_SEPARATION_KM = 0.2                # two distinct ZIPs never share a centroid
MAX_STEP_REGULARITY = 0.05       # spread of the gaps between consecutive coordinates; ~0 means an arithmetic sequence


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) pairs."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def in_bbox(lat: float, lon: float) -> bool:
    la0, la1, lo0, lo1 = DFW_BBOX
    return la0 <= lat <= la1 and lo0 <= lon <= lo1


def validate_centroids(obj) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """Untrusted {zip: [lat, lon]} -> (clean centroids, errors). Pure function, used by the build script and the loader."""
    out, errors = {}, []
    if not isinstance(obj, dict):
        return {}, ["centroids must be a JSON object {zip: [lat, lon]}"]
    for z, v in obj.items():
        zs = str(z).strip()
        if not (len(zs) == 5 and zs.isdigit()):
            errors.append(f"{z!r} is not a 5-digit ZIP")
            continue
        try:
            lat, lon = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError, KeyError):
            errors.append(f"{zs}: centroid is not a [lat, lon] pair ({v!r})")
            continue
        if not in_bbox(lat, lon):
            errors.append(f"{zs}: ({lat}, {lon}) is outside the DFW bounding box {DFW_BBOX}")
            continue
        out[zs] = (lat, lon)
    return out, errors


def probe_errors(centroids: dict[str, tuple[float, float]], tolerance_km: float = PROBE_TOLERANCE_KM) -> list[str]:
    """Probe ZIPs in `centroids` that sit further than `tolerance_km` from their known position."""
    out = []
    for z, truth in sorted(PROBES.items()):
        if z in centroids:
            d = haversine_km(centroids[z], truth)
            if d > tolerance_km:
                out.append(f"{z} is {d:.0f} km from its true centroid {truth} (limit {tolerance_km:.0f} km)")
    return out


def _step_variation(values: list[float]) -> float | None:
    """Spread of the gaps between consecutive values, relative to their size. ~0 means a constant step, i.e. an
    arithmetic sequence. None when the values do not move at all (that is reported separately)."""
    diffs = [b - a for a, b in zip(values, values[1:])]
    scale = sum(abs(d) for d in diffs) / len(diffs) if diffs else 0.0
    if len(diffs) < 3 or scale < 1e-9:
        return None
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))
    return sd / scale


def degeneracy_report(centroids: dict[str, tuple[float, float]]) -> list[str]:
    """Signs that the coordinates were invented from the ZIP numbers rather than recalled. Real centroids scatter;
    an interpolated sequence does not. Three signatures, all seen in practice (AI_USE.md):
      * a coordinate that never changes across every ZIP;
      * a coordinate that walks in a constant step through the ZIPs in sorted order (an arithmetic sequence);
      * two distinct ZIPs on the same point.
    Empty list = nothing suspicious."""
    items = sorted(centroids.items())
    lines = []
    for axis, name in ((0, "latitude"), (1, "longitude")):
        vals = [c[axis] for _, c in items]
        if len(vals) >= 3 and max(vals) - min(vals) < 1e-9:
            lines.append(f"every ZIP has the same {name} {vals[0]}: interpolated, not recalled")
            continue
        cv = _step_variation(vals)
        if cv is not None and cv < MAX_STEP_REGULARITY:
            lines.append(f"{name} walks in a constant step through the ZIPs (step variation {cv:.3f}): "
                         "interpolated, not recalled")
    dup = [f"{z1} and {z2} share a centroid {c1}"
           for i, (z1, c1) in enumerate(items) for z2, c2 in items[i + 1:] if haversine_km(c1, c2) < MIN_SEPARATION_KM]
    return lines + dup[:8]


def load_centroids(path=CENTROIDS_PATH) -> dict[str, tuple[float, float]]:
    """Centroids shipped with the snapshot; {} when the file is absent or unreadable (callers fall back to numeric)."""
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        return validate_centroids(json.loads(p.read_text()).get("centroids", {}))[0]
    except (OSError, ValueError):
        return {}


def numeric_order(zips: list[str]) -> dict[str, list[str]]:
    """Fallback ordering: the numerically nearest ZIP codes (ties broken by the ZIP itself, so it is deterministic)."""
    return {z: sorted((o for o in zips if o != z), key=lambda o: (abs(int(o) - int(z)), int(o))) for z in zips}


def neighbour_order(zips: list[str], centroids: dict[str, tuple[float, float]] | None = None) -> dict[str, list[str]]:
    """{zip: other ZIPs, nearest first}. Great-circle distance where both centroids are known, numeric distance
    otherwise; ZIPs with a centroid always come before ZIPs without one, and the ZIP number breaks every tie."""
    centroids = centroids or {}
    zips = list(zips)
    out = numeric_order(zips)
    for z in zips:
        if z not in centroids:
            continue
        known = [o for o in zips if o != z and o in centroids]
        unknown = [o for o in zips if o != z and o not in centroids]
        known.sort(key=lambda o: (haversine_km(centroids[z], centroids[o]), int(o)))
        unknown.sort(key=lambda o: (abs(int(o) - int(z)), int(o)))
        out[z] = known + unknown
    return out
