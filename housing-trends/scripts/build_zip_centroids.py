"""Ask the LLM ONCE for the approximate centroid of every ZIP in the snapshot, validate it hard, and freeze it to
data/snapshot/zip_centroids.json.

  python scripts/build_zip_centroids.py                 # real LLM (needs a key in .env.local); replies are cached
  python scripts/build_zip_centroids.py --show          # print what is currently frozen, change nothing
  python scripts/build_zip_centroids.py --audit         # re-validate the frozen file, change nothing

This is a BUILD step, not a run-time step: the pipeline only ever reads the frozen file (housing_trends/geo.py), so
results stay reproducible without an API key.

WHY THE VALIDATION IS THIS HEAVY (see AI_USE.md). Asked for all 42 ZIPs in one call, the model returned a FABRICATED
answer that passed a bounding-box check: latitude pinned at 32.95 and longitude marching in exact -0.03 steps in ZIP
order, with the last seven ZIPs on one identical point. It had interpolated from the ZIP number, which is precisely
what the prompt forbade. Three defences, all in code:
  * small batches, so the model answers each ZIP concretely instead of extrapolating a sequence;
  * PROBE ZIPs: every batch secretly contains ZIPs whose true centroid is known here. A batch whose probes are wrong
    is rejected whole, because a model that misplaces a ZIP it should know cannot be trusted for the others;
  * a degeneracy check on the assembled set: duplicate points, and coordinates that are a linear function of the ZIP
    number, are rejected.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from housing_trends.geo import (CENTROIDS_PATH, DFW_BBOX, degeneracy_report, haversine_km, load_centroids,
                                neighbour_order, probe_errors, validate_centroids)
from housing_trends.llm import CachedLLM, llm_from_env
from housing_trends.parser_agent import _extract_json as extract_json

BATCH = 6               # ZIPs asked for at once (plus the probes); small batches stop the model extrapolating
PROBES_PER_BATCH = 2

SYSTEM_PROMPT = f"""You are a geographic reference for the Dallas-Fort Worth metropolitan area, Texas, USA.
For every ZIP code given, return its approximate centroid as decimal degrees [latitude, longitude].
Every centroid must lie inside the bounding box latitude {DFW_BBOX[0]}..{DFW_BBOX[1]}, longitude {DFW_BBOX[2]}..{DFW_BBOX[3]}.
Recall where each ZIP code actually is on the ground. NEVER interpolate or extrapolate a centroid from the numeric
value of the ZIP code: neighbouring numbers are often far apart, and two different ZIPs never share a centroid.
If you do not know a ZIP code, omit it rather than guessing a position from its number.
Reply with ONE JSON object and nothing else: {{"centroids": {{"75201": [32.7815, -96.7954], ...}}}}"""


def ask_batch(llm, zips, probes, attempts=2):
    """One batch: LLM proposes, code validates shape, bbox and the hidden probes. Returns {} if the batch is untrusted."""
    asked = sorted(set(zips) | set(probes))
    feedback = ""
    for attempt in range(1, attempts + 1):
        try:
            obj = extract_json(llm(SYSTEM_PROMPT, "ZIPS:\n" + json.dumps(asked) + feedback))
        except Exception as ex:
            feedback = f"\n\nYour previous reply was not valid JSON ({ex}). Reply with ONLY the JSON object."
            continue
        clean, errors = validate_centroids(obj.get("centroids", obj))
        bad = probe_errors(clean)
        if bad:
            print(f"    attempt {attempt}: REJECTED, {len(bad)} probe(s) misplaced: " + "; ".join(bad[:3]))
            feedback = ("\n\nYour previous reply placed these ZIP codes far from their true location: "
                        + "; ".join(bad) + ". Recall each ZIP's real position and answer again.")
            continue
        if errors:
            print(f"    attempt {attempt}: {len(errors)} entry(ies) rejected: {errors[0]}")
        return {z: c for z, c in clean.items() if z in zips}
    return {}


def ask(llm, zips, probe_zips):
    good = {}
    batches = [zips[i:i + BATCH] for i in range(0, len(zips), BATCH)]
    for i, batch in enumerate(batches, 1):
        probes = [probe_zips[(i * PROBES_PER_BATCH + j) % len(probe_zips)] for j in range(PROBES_PER_BATCH)]
        got = ask_batch(llm, batch, probes)
        good.update(got)
        print(f"  batch {i}/{len(batches)}: {len(got)}/{len(batch)} accepted")
    return good


def report(cen, zips):
    print(f"\n{len(cen)}/{len(zips)} ZIPs have a centroid")
    for line in degeneracy_report(cen):
        print(f"  DEGENERACY: {line}")
    for line in probe_errors(cen):
        print(f"  PROBE: {line}")
    order, num = neighbour_order(zips, cen), neighbour_order(zips, {})
    changed = [z for z in zips if order[z][:2] != num[z][:2]]
    print(f"{len(changed)}/{len(zips)} ZIPs get different 2 nearest neighbours than the numeric ordering")
    for z in changed[:10]:
        print(f"  {z}: numeric {', '.join(num[z][:2])}  ->  geographic {', '.join(order[z][:2])}")
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(ROOT / "data/snapshot/listings_dfw.csv"))
    ap.add_argument("--show", action="store_true", help="print the frozen file and exit")
    ap.add_argument("--audit", action="store_true", help="re-validate the frozen file and exit")
    a = ap.parse_args()

    zips = sorted(pd.read_csv(a.snapshot, dtype={"zip_code": str}).zip_code.dropna().unique())
    if a.show or a.audit:
        cen = load_centroids()
        if a.show:
            for z in zips:
                print(f"  {z}: {cen.get(z, '(none)')}")
        bad = degeneracy_report(cen) + probe_errors(cen)
        report(cen, zips)
        sys.exit(1 if bad else 0)

    base = llm_from_env()
    llm = CachedLLM(base, ROOT / "results/llm_cache", tag=base.name)
    from housing_trends.geo import PROBES
    probe_zips = [z for z in PROBES if z in zips]
    print(f"asking {base.name} for {len(zips)} ZIP centroids in batches of {BATCH} (+{PROBES_PER_BATCH} hidden probes)")
    cen = ask(llm, zips, probe_zips)

    problems = degeneracy_report(cen) + probe_errors(cen)
    order = report(cen, zips)
    if problems or not cen:
        sys.exit("\nREFUSING to write: the reply did not survive validation. The frozen file was left untouched.")

    payload = {"source": f"LLM {base.name} via scripts/build_zip_centroids.py",
               "validation": f"bbox {list(DFW_BBOX)}, {len(probe_zips)} probe ZIPs with known centroids, "
                             "duplicate-point and ZIP-number-linearity checks",
               "bbox": list(DFW_BBOX), "n_zips": len(cen), "missing": [z for z in zips if z not in cen],
               "centroids": {z: [round(cen[z][0], 4), round(cen[z][1], 4)] for z in sorted(cen)}}
    pathlib.Path(CENTROIDS_PATH).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {CENTROIDS_PATH}: {len(cen)}/{len(zips)} ZIPs ({len(payload['missing'])} fall back to numeric order)")


if __name__ == "__main__":
    main()
