#!/usr/bin/env python
"""Build the FROZEN synthetic Dallas-Fort Worth snapshot, grounded in two real datasets.

Why synthetic: the Kaggle file's usable dates cover only ~6 months (Nov 2021 - Apr 2022), so a real
12-month trend is impossible. Zillow ZHVI gives the missing time dimension.

How it is grounded
  * LEVEL + TREND  : synthetic price = f_zip x ZHVI(city, month) x size-adjustment x noise.
                     ZHVI supplies the month-to-month movement for the chosen city.
  * f_zip          : calibrated on REAL Kaggle sold rows (Nov21-Apr22): median of price / ZHVI(city, sold month),
                     shrunk toward the metro value for ZIPs with few rows. It absorbs both the ZIP's
                     price level relative to its city AND the gap between a typical-home index and sale prices.
  * PROPERTY MIX   : bed / bath / house_size / acre_lot are bootstrapped from REAL rows of the same ZIP
                     (missing values stay missing, as in the real data).
  * SIZE ELASTICITY, NOISE: estimated from the real rows (within-ZIP regression; robust MAD).
  * VOLUME         : per-ZIP monthly Poisson rate from real counts (gamma-mixed for clumpiness), so
                     thin ZIP-months arise naturally -> the coverage loop has something to do.
  * INJECTED FAULTS (labelled in a SEPARATE truth file the agents must never read):
        typo_x10 / typo_div10 : bad listings (price off by 10x)
        rural_lowvalue        : legitimately cheap big-lot / small-house listings
    These exercise Pooling's two outlier hypotheses ("bad listing" vs "legitimate rural plot").

Everything is seeded; same inputs + seed => byte-identical outputs (see manifest sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DFW_METRO = "Dallas-Fort Worth-Arlington, TX"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- inputs
def load_zhvi(path):
    z = pd.read_csv(path)
    z = z[(z.State == "TX") & (z.Metro == DFW_METRO)]
    mcols = [c for c in z.columns if c[:2] == "20"]
    long = z.melt(id_vars=["RegionName", "CountyName"], value_vars=mcols, var_name="date", value_name="zhvi")
    long["month"] = long["date"].str[:7]
    long["city_key"] = long.RegionName.str.lower().str.strip()
    long = long.drop(columns="date").rename(columns={"RegionName": "city", "CountyName": "county"})
    nun = long.groupby("city_key").county.nunique()
    ambiguous = sorted(nun[nun > 1].index)          # same city name in 2 counties: cannot match Kaggle safely
    return long[~long.city_key.isin(ambiguous)].copy(), ambiguous


def load_kaggle(path, calib_start, calib_end):
    use = ["status", "price", "bed", "bath", "acre_lot", "city", "state", "zip_code", "house_size", "prev_sold_date"]
    parts = []
    for ch in pd.read_csv(path, usecols=use, dtype={"zip_code": str}, chunksize=300_000):
        parts.append(ch[(ch.state == "Texas") & (ch.status == "sold")])
    k = pd.concat(parts)
    k["city_key"] = k.city.astype(str).str.lower().str.strip()
    k["zip"] = k.zip_code.str.extract(r"(\d{5})")[0]
    k["month"] = pd.to_datetime(k.prev_sold_date, errors="coerce").dt.strftime("%Y-%m")
    k = k[(k.month >= calib_start) & (k.month <= calib_end) & (k.price > 0) & k.zip.notna()]
    return k.copy()


# ---------------------------------------------------------------- calibration
def calibrate(k, zhvi, shrink_k):
    k = k.merge(zhvi[["city_key", "month", "zhvi"]], on=["city_key", "month"], how="inner").dropna(subset=["zhvi"])
    k["lratio"] = np.log(k.price / k.zhvi)
    size_ok = (k.house_size >= 300) & (k.house_size <= 10_000)
    k["lsize"] = np.log(k.house_size.where(size_ok))

    s = k.dropna(subset=["lsize"])                   # size elasticity: within-ZIP OLS, extremes trimmed
    dm = s.lsize - s.groupby("zip").lsize.transform("mean")
    dy = s.lratio - s.groupby("zip").lratio.transform("mean")
    keep = dy.abs() <= dy.abs().quantile(0.98)
    beta = float(np.clip((dm[keep] * dy[keep]).sum() / (dm[keep] ** 2).sum(), 0.3, 1.2))

    metro_size = float(k.house_size[size_ok].median())
    k["zip_size_ref"] = k.groupby("zip").house_size.transform(lambda x: x[(x >= 300) & (x <= 10_000)].median())
    k["zip_size_ref"] = k.zip_size_ref.fillna(metro_size)
    k["adj"] = k.lratio - np.where(k.lsize.notna(), beta * (k.lsize - np.log(k.zip_size_ref)), 0.0)

    f_metro = float(k.adj.median())
    g = k.groupby("zip")
    n = g.size()
    f_zip_raw = g.adj.median()
    f_log = (n * f_zip_raw + shrink_k * f_metro) / (n + shrink_k)
    resid = k.adj - k.zip.map(f_zip_raw)
    sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))   # robust: ignores real outliers

    zip_city = k.groupby("zip").city_key.agg(lambda s: s.mode().iat[0])
    n_months = k.month.nunique()
    zips = {}
    for z, df in g:
        zips[z] = dict(n=int(len(df)), f_log=float(f_log[z]), city_key=zip_city[z],
                       lam=len(df) / n_months, size_ref=float(df.zip_size_ref.iat[0]),
                       city=df.city.mode().iat[0],
                       bed=df.bed.to_numpy(float), bath=df.bath.to_numpy(float),
                       acre=df.acre_lot.to_numpy(float), size=df.house_size.to_numpy(float))
    return dict(beta=beta, sigma=sigma, f_metro=f_metro, n_rows=int(len(k)), n_months=int(n_months)), zips


# ---------------------------------------------------------------- generation
def generate(zips, zhvi, months, p, rng):
    z = zhvi.set_index(["city_key", "month"]).zhvi
    out = []
    for zp in sorted(zips):
        info = zips[zp]
        for m in months:
            lvl = np.exp(info["f_log"]) * z[(info["city_key"], m)]
            n = rng.poisson(info["lam"] * p["volume_scale"] * rng.gamma(4.0, 0.25))
            if n == 0:
                continue
            i = rng.integers(0, len(info["size"]), n)
            bed, bath, acre, size = info["bed"][i].copy(), info["bath"][i].copy(), info["acre"][i].copy(), info["size"][i].copy()
            size = np.round(size * np.exp(rng.normal(0, 0.05, n)))            # NaN stays NaN
            adj = np.where(np.isnan(size), 0.0, p["beta"] * (np.log(np.clip(size, 1, None)) - np.log(info["size_ref"])))
            eps = rng.normal(0, p["sigma"], n)
            price = lvl * np.exp(adj + eps)
            fault = np.full(n, "", dtype=object)
            u = rng.random(n)
            typo = u < p["typo_rate"]
            up = rng.random(n) < 0.5
            price = np.where(typo & up, price * 10, np.where(typo & ~up, price / 10, price))
            fault[typo & up], fault[typo & ~up] = "typo_x10", "typo_div10"
            rur = (u >= p["typo_rate"]) & (u < p["typo_rate"] + p["rural_rate"])
            r = int(rur.sum())
            if r:
                acre[rur] = np.exp(rng.uniform(np.log(5), np.log(40), r)).round(2)
                size[rur] = rng.integers(600, 1100, r)
                bed[rur], bath[rur] = 2, 1
                price[rur] = lvl * rng.uniform(0.25, 0.45, r) * np.exp(rng.normal(0, p["sigma"] * 0.5, r))
                fault[rur] = "rural_lowvalue"
            dim = pd.Period(m).days_in_month
            out.append(pd.DataFrame(dict(
                status="sold", price=np.round(price, -2), bed=bed, bath=bath, acre_lot=acre,
                city=info["city"], state="Texas", zip_code=zp, house_size=size,
                sold_date=[f"{m}-{d:02d}" for d in rng.integers(1, dim + 1, n)], month=m, _fault=fault)))
    df = pd.concat(out, ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)             # no ordering by ZIP/month
    df.insert(0, "listing_id", np.arange(1, len(df) + 1))
    return df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle", default=str(ROOT / "data/raw/realtor-data.zip.csv"))
    ap.add_argument("--zhvi-city", default=str(ROOT / "data/raw/City_zhvi_uc_sfrcondo_tier_0_33_0_67_sm_sa_month.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/snapshot"))
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--end-month", default=None, help="last window month; default = last month in ZHVI")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--calib-start", default="2021-11")
    ap.add_argument("--calib-end", default="2022-04")
    ap.add_argument("--min-zip-rows", type=int, default=10, help="min real rows for a ZIP to be included")
    ap.add_argument("--shrink-k", type=float, default=20.0)
    ap.add_argument("--volume-scale", type=float, default=1.0)
    ap.add_argument("--typo-rate", type=float, default=0.004)
    ap.add_argument("--rural-rate", type=float, default=0.003)
    a = ap.parse_args()

    zhvi, ambiguous = load_zhvi(a.zhvi_city)
    end = a.end_month or zhvi.month.max()
    months = [str(x) for x in pd.period_range(end=end, periods=a.window, freq="M")]
    prev = str(pd.Period(months[0]) - 1)

    k = load_kaggle(a.kaggle, a.calib_start, a.calib_end)
    cal, zips = calibrate(k, zhvi, a.shrink_k)

    need = set(months) | {prev}
    full = zhvi[zhvi.month.isin(need)].groupby("city_key").zhvi.apply(lambda s: s.notna().sum() == len(need))
    ok_cities = set(full[full].index)
    n_all = len(zips)
    zips = {z: v for z, v in zips.items() if v["n"] >= a.min_zip_rows and v["city_key"] in ok_cities}

    p = dict(beta=cal["beta"], sigma=cal["sigma"], volume_scale=a.volume_scale,
             typo_rate=a.typo_rate, rural_rate=a.rural_rate)
    rng = np.random.default_rng(a.seed)
    df = generate(zips, zhvi, months, p, rng)

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["listing_id", "status", "price", "bed", "bath", "acre_lot", "house_size",
            "city", "state", "zip_code", "sold_date", "month"]
    snap = df[cols].copy()
    snap["bed"] = snap.bed.round().astype("Int64")
    snap["price"] = snap.price.astype("int64")
    snap["house_size"] = snap.house_size.astype("Int64")
    snap.to_csv(out / "listings_dfw.csv", index=False)
    df[["listing_id", "_fault"]].rename(columns={"_fault": "fault"}).to_csv(out / "listings_dfw_truth.csv", index=False)

    used = {v["city_key"] for v in zips.values()}
    zs = zhvi[zhvi.city_key.isin(used) & zhvi.month.isin(need)][["city", "county", "city_key", "month", "zhvi"]]
    zs.sort_values(["city_key", "month"]).to_csv(out / "zhvi_dfw_city_long.csv", index=False)

    manifest = dict(
        seed=a.seed, window=months, calibration_window=[a.calib_start, a.calib_end],
        args={k_: v for k_, v in vars(a).items() if k_ not in ("kaggle", "zhvi_city", "out_dir")},
        calibration=dict(beta_size_elasticity=round(cal["beta"], 4), sigma_log_noise=round(cal["sigma"], 4),
                         f_metro=round(cal["f_metro"], 4), real_rows_used=cal["n_rows"],
                         zips_calibrated=n_all, zips_in_snapshot=len(zips), ambiguous_zhvi_cities_dropped=ambiguous),
        output=dict(rows=len(snap), zips=int(snap.zip_code.nunique()), cities=int(snap.city.nunique()),
                    faults=df._fault.replace("", "none").value_counts().to_dict()),
        input_sha256={"kaggle": sha256(a.kaggle), "zhvi_city": sha256(a.zhvi_city)},
        output_sha256={f: sha256(out / f) for f in ("listings_dfw.csv", "listings_dfw_truth.csv", "zhvi_dfw_city_long.csv")},
        numpy=np.__version__, pandas=pd.__version__)
    (out / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest["calibration"], indent=2, default=str))
    print(json.dumps(manifest["output"], indent=2, default=str))


if __name__ == "__main__":
    main()