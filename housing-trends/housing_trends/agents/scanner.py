"""Scanner Agent: retrieve matching listings per ZIP-month bucket and report, for each bucket, whether and how
widening the scope could surface more listings.

ONE LLM DECISION, NO LLM ARITHMETIC: which neighbouring ZIPs are comparable markets to pool from.
  * Geography alone picks poor neighbours: a ZIP next door can be a much richer or much cheaper market, and pooling
    it distorts the median. So, once per run and only for ZIPs that have thin buckets, code shows the LLM the
    N_CANDIDATES geographically nearest ZIPs with their median price, house size and listing count, and the LLM
    ranks the most comparable ones (rank_neighbours).
  * Code validates the reply (only listed candidates, no duplicates, at most MAX_ZIP_EXTRA) and falls back to the
    geographic order for any ZIP without a valid ranking. The ranking only reorders neighbours; every count and gain
    below is still computed by code, exactly, from that order.
  * The stand-in (rule_based_neighbours) keeps the geographic order, so `--llm offline` scans exactly as before.

WIDEN ONLY WHAT IS THIN: a bucket whose own listings (all user filters, no pooling) already reach
`min_listings` is never pooled or relaxed, whatever the scope says. Only thin buckets receive the scope.
Otherwise a rescan meant to fix thin buckets would silently change healthy ones.

Per-bucket report (one row per requested ZIP x month in the criteria window):
  widened              True if this bucket was thin at base and therefore received the scope
  zip_extra_used, time_pad_used   widening depth actually applied to this bucket (0 if not widened)
  n_base               own listings under ALL user filters, no pooling (decides `widened`; never changes)
  n_own, n, n_pooled   listings from the bucket itself / in the current scope / pooled in from neighbours
  gain_time            extra listings if the date window were widened to the MAX allowed padding
  gain_zip             extra listings if the ZIP radius were widened to the MAX allowed neighbours
  gain_filters         extra listings if every still-relaxable (set, non-protected) filter were dropped
  potential            extra listings if ALL moves were applied together (upper bound)
  moves                which of {time, zip, filters} would add >= 1 listing  (Pooling picks from these)
  exhaustive           potential == 0: no allowed widening can surface another listing  -> Pooling must stop
  reachable            n + potential >= min_listings: the Critic's threshold is attainable at all
  thin                 n < min_listings
"Gain" is exact (it equals what the corresponding rescan would add); tests check this by brute force.

Assumptions (document in the report):
  * "Adjacent ZIPs" = the numerically nearest ZIP codes in the snapshot (no coordinates available).
  * A listing with a missing value for an ACTIVE filter's field (e.g. bed) does not match that filter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..context import month_index, month_range
from ..criteria import Criteria
from ..geo import load_centroids, neighbour_order
from ..state import MAX_TIME_PAD, MAX_ZIP_EXTRA, MIN_LISTINGS, Scope

OPTIONAL_COLS = ("bed", "bath", "house_size", "acre_lot", "property_type")


# ---------------------------------------------------------------- snapshot
class Snapshot:
    def __init__(self, df: pd.DataFrame, months=None, zips=None, centroids=None):
        """`months` / `zips` declare the coverage explicitly; by default it is inferred from the rows.
        `centroids` {zip: (lat, lon)} orders the pooling neighbours geographically; without it (the default, and what
        the unit tests use) the numerically nearest ZIP codes are used. See housing_trends.geo."""
        need = {"listing_id", "zip_code", "month", "price"} - set(df.columns)
        if need:
            raise ValueError(f"snapshot is missing columns: {sorted(need)}")
        df = df.copy()
        df["zip_code"] = df.zip_code.astype(str).str.extract(r"(\d{5})")[0]
        df["month"] = df.month.astype(str).str[:7]
        for c in OPTIONAL_COLS:
            if c not in df.columns:
                df[c] = np.nan
        self.df = df
        self.zips = sorted(set(zips) if zips else df.zip_code.dropna().unique(), key=int)
        self.months = list(months) if months else month_range(df.month.min(), df.month.max())
        self.centroids = dict(centroids or {})
        self._order = neighbour_order(self.zips, self.centroids)

    @classmethod
    def from_csv(cls, path, centroids=None) -> "Snapshot":
        return cls(pd.read_csv(path, dtype={"zip_code": str}), centroids=centroids)

    @classmethod
    def frozen(cls, df, **kw) -> "Snapshot":
        """The snapshot as the application uses it: pooling neighbours ordered by the frozen ZIP centroids."""
        return cls(df, centroids=load_centroids(), **kw)

    def neighbours(self, z: str, k: int) -> list[str]:
        return self._order[z][:k]

    def profiles(self) -> dict[str, dict]:
        """{zip: median price, median house size, listing count, city} over the whole snapshot (computed once)."""
        if getattr(self, "_profiles", None) is None:
            g = self.df.groupby("zip_code")
            price, size, n = g.price.median(), g.house_size.median(), g.size()
            city = (g.city.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
                    if "city" in self.df.columns else {})
            self._profiles = {z: {"median_price": int(price[z]) if z in price and pd.notna(price[z]) else None,
                                  "median_sqft": int(size[z]) if z in size and pd.notna(size[z]) else None,
                                  "listings": int(n.get(z, 0)), "city": city.get(z) if len(city) else None}
                              for z in self.zips}
        return self._profiles


# ---------------------------------------------------------------- filters
def active_filters(c: Criteria) -> list[str]:
    on = {"price": c.price_min is not None or c.price_max is not None,
          "beds": c.beds_min is not None or c.beds_max is not None,
          "baths": c.baths_min is not None,
          "sqft": c.sqft_min is not None or c.sqft_max is not None,
          "property_type": bool(c.property_types)}
    return [f for f in ("price", "beds", "baths", "sqft", "property_type") if on[f]]


def relaxable_filters(c: Criteria, scope: Scope) -> list[str]:
    """Set, NON-protected filters that Pooling has not relaxed yet."""
    return [f for f in active_filters(c) if f not in c.protected_filters and f not in scope.relaxed]


def filter_mask(df: pd.DataFrame, c: Criteria, filters: list[str]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if "price" in filters:
        if c.price_min is not None: m &= df.price >= c.price_min
        if c.price_max is not None: m &= df.price <= c.price_max
    if "beds" in filters:
        if c.beds_min is not None: m &= df.bed >= c.beds_min
        if c.beds_max is not None: m &= df.bed <= c.beds_max
    if "baths" in filters and c.baths_min is not None:
        m &= df.bath >= c.baths_min
    if "sqft" in filters:
        if c.sqft_min is not None: m &= df.house_size >= c.sqft_min
        if c.sqft_max is not None: m &= df.house_size <= c.sqft_max
    if "property_type" in filters and c.property_types:
        m &= df.property_type.astype(str).str.lower().isin(c.property_types)
    return m                                           # NaN comparisons are False -> not a match


# ---------------------------------------------------------------- scan
@dataclass
class ScanResult:
    criteria_key: str
    scope: Scope
    buckets: pd.DataFrame        # one row per (zip_code, month); see module docstring
    members: pd.DataFrame        # (zip_code, month, listing_id, own): listings behind each bucket

    def thin(self) -> pd.DataFrame:
        return self.buckets[self.buckets.thin]

    def summary(self) -> str:
        b = self.buckets
        return (f"{len(b)} buckets, {int(b.widened.sum())} widened, {int(b.thin.sum())} thin (<min), {int(b.exhaustive.sum())} exhaustive, "
                f"{int((b.thin & ~b.reachable).sum())} thin&unreachable | depth used: time_pad max {int(b.time_pad_used.max())}, "
                f"zip_extra max {int(b.zip_extra_used.max())}, relaxed={list(self.scope.relaxed)}")


def _count_matrix(df, zips, months):
    g = df.groupby(["zip_code", "month"]).size().unstack(fill_value=0)
    return g.reindex(index=zips, columns=months, fill_value=0).to_numpy()


def scan(c: Criteria, scope: Scope, snap: Snapshot, min_listings: int = MIN_LISTINGS,
         order: dict[str, list[str]] | None = None) -> ScanResult:
    """`order` {zip: neighbours, best first} overrides the snapshot's geographic order for the ZIPs it names."""
    order = order or {}

    def nbrs(z: str, k: int) -> list[str]:
        return order[z][:k] if z in order else snap.neighbours(z, k)

    illegal = sorted(set(scope.relaxed) & set(c.protected_filters))
    if illegal:
        raise ValueError(f"scope relaxes protected filter(s) {illegal}; Pooling may only relax non-protected ones")
    missing = [z for z in c.zips if z not in snap.zips]
    if missing:
        raise ValueError(f"criteria ZIPs not in snapshot: {missing}")
    window = month_range(c.start_month, c.end_month)
    if month_index(window[0]) < month_index(snap.months[0]) or month_index(window[-1]) > month_index(snap.months[-1]):
        raise ValueError(f"window {window[0]}..{window[-1]} outside snapshot {snap.months[0]}..{snap.months[-1]}")

    act = active_filters(c)
    cur = [f for f in act if f not in scope.relaxed]                          # filters in force for widened buckets
    all_relaxed = [f for f in cur if f in c.protected_filters]                # what remains if everything relaxable is dropped
    df = snap.df
    m_base, m_cur = filter_mask(df, c, act), filter_mask(df, c, cur)
    A_base = _count_matrix(df[m_base], snap.zips, snap.months)
    A_cur = _count_matrix(df[m_cur], snap.zips, snap.months)
    A_rel = _count_matrix(df[filter_mask(df, c, all_relaxed)], snap.zips, snap.months)
    zidx, midx = {z: i for i, z in enumerate(snap.zips)}, {m: i for i, m in enumerate(snap.months)}
    nm = len(snap.months)

    def pooled(A, z, mi, extra, pad):
        rows = [zidx[z]] + [zidx[o] for o in nbrs(z, extra)]
        return int(A[np.ix_(rows, range(max(0, mi - pad), min(nm - 1, mi + pad) + 1))].sum())

    rows, plan = [], []
    for z in c.zips:
        for m in window:
            mi = midx[m]
            n_base = int(A_base[zidx[z], mi])
            widen = n_base < min_listings                                    # healthy buckets keep the default scope
            ze, tp = scope.depth(z, m) if widen else (0, 0)
            A = A_cur if widen else A_base
            n = pooled(A, z, mi, ze, tp)
            g_t = pooled(A, z, mi, ze, MAX_TIME_PAD) - n
            g_z = pooled(A, z, mi, MAX_ZIP_EXTRA, tp) - n
            g_f = pooled(A_rel, z, mi, ze, tp) - n
            pot = pooled(A_rel, z, mi, MAX_ZIP_EXTRA, MAX_TIME_PAD) - n
            n_own = int(A[zidx[z], mi])
            plan.append((z, m, widen, ze, tp))
            rows.append(dict(zip_code=z, month=m, widened=widen, zip_extra_used=ze, time_pad_used=tp,
                             n_base=n_base, n_own=n_own, n=n, n_pooled=n - n_own,
                             gain_time=g_t, gain_zip=g_z, gain_filters=g_f, potential=pot,
                             moves=",".join(k for k, g in (("time", g_t), ("zip", g_z), ("filters", g_f)) if g > 0),
                             exhaustive=pot == 0, reachable=n + pot >= min_listings, thin=n < min_listings))
    buckets = pd.DataFrame(rows)

    ids_cur = {k: v.to_numpy() for k, v in df[m_cur].groupby(["zip_code", "month"]).listing_id}
    ids_base = {k: v.to_numpy() for k, v in df[m_base].groupby(["zip_code", "month"]).listing_id}
    mem = []
    for z, m, widen, ze, tp in plan:
        ids, mi = (ids_cur if widen else ids_base), midx[m]
        for zz in [z] + nbrs(z, ze):
            for mm in snap.months[max(0, mi - tp): min(nm - 1, mi + tp) + 1]:
                for lid in ids.get((zz, mm), ()):
                    mem.append((z, m, int(lid), zz == z and mm == m))
    members = (pd.DataFrame(mem, columns=["zip_code", "month", "listing_id", "own"])
               .sort_values(["zip_code", "month", "listing_id"]).reset_index(drop=True))
    return ScanResult(c.key(), scope, buckets, members)


# ---------------------------------------------------------------- neighbour ranking (the Scanner's LLM decision)
MARK = "NEIGHBOURS_JSON:\n"
N_CANDIDATES = MAX_ZIP_EXTRA + 4      # geographically nearest ZIPs shown to the model per target
RANK_BATCH = 4                        # targets per prompt: a 7B model drops or mixes up targets in larger batches

SYSTEM_PROMPT = f"""You are the market-comparability analyst of a housing price-trend system.
When a ZIP code has too few listings in a month, listings from other ZIP codes are pooled in. Pooling from a ZIP that
is a different market (much more or much less expensive, very different house sizes) distorts the median price.
NEIGHBOURS_JSON lists "targets". Each target has its own profile (median_price, median_sqft, listings, city) and
"candidates": nearby ZIPs, nearest first (proximity_rank 1 = nearest), each with the same profile plus
price_ratio = candidate median_price / target median_price.
For EVERY target choose up to {MAX_ZIP_EXTRA} of its candidates that are the most comparable markets, most comparable
first. Prefer price_ratio close to 1 and a similar median_sqft; between similar markets prefer the nearer one.
Use only candidate ZIPs listed for that target.
Reply with ONE JSON object and nothing else:
{{"rankings":[{{"zip_code":"<target>","comparable":["<zip>", "..."],"reason":"<short>"}}]}}"""


def thin_zips(c: Criteria, snap: Snapshot, min_listings: int = MIN_LISTINGS) -> list[str]:
    """Requested ZIPs with at least one bucket below the threshold under ALL user filters (the only ZIPs whose
    neighbours can ever be pooled). Independent of the scope, so it is decided once per run."""
    window = [m for m in month_range(c.start_month, c.end_month) if m in snap.months]
    A = _count_matrix(snap.df[filter_mask(snap.df, c, active_filters(c))], list(c.zips), window)
    return [z for z, row in zip(c.zips, A) if (row < min_listings).any()]


def candidates(snap: Snapshot, z: str) -> list[str]:
    return snap.neighbours(z, N_CANDIDATES)


def build_neighbour_prompt(targets: list[str], snap: Snapshot) -> str:
    prof = snap.profiles()
    body = []
    for z in targets:
        base = prof[z]["median_price"]
        cands = []
        for rank, o in enumerate(candidates(snap, z), 1):
            p = prof[o]
            ratio = round(p["median_price"] / base, 2) if base and p["median_price"] else None
            cands.append({"zip_code": o, "proximity_rank": rank, "price_ratio": ratio, **p})
        body.append({"zip_code": z, **prof[z], "candidates": cands})
    return MARK + json.dumps({"max_per_target": MAX_ZIP_EXTRA, "targets": body}, sort_keys=True)


def validate_rankings(obj, allowed: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str], list[str]]:
    """-> ({target: comparable ZIPs}, {target: reason}, errors). Valid entries are kept even when others are rejected."""
    picks, reasons, errors = {}, {}, []
    if not isinstance(obj.get("rankings"), list):
        return {}, {}, ["reply must contain a list 'rankings'"]
    for r in obj["rankings"]:
        try:
            z, comp = str(r["zip_code"]), r["comparable"]
        except (KeyError, TypeError):
            errors.append(f"malformed ranking entry: {r!r}")
            continue
        if z not in allowed:
            errors.append(f"{z!r} is not one of the targets {sorted(allowed)}")
            continue
        comp = [str(o) for o in comp] if isinstance(comp, list) else None
        bad = None if comp is None else [o for o in comp if o not in allowed[z]]
        if not comp:
            errors.append(f"{z}: 'comparable' must be a non-empty list of candidate ZIPs")
        elif bad:
            errors.append(f"{z}: {bad} are not candidates of {z} (candidates: {allowed[z]})")
        elif len(set(comp)) != len(comp):
            errors.append(f"{z}: 'comparable' repeats a ZIP")
        elif len(comp) > MAX_ZIP_EXTRA:
            errors.append(f"{z}: at most {MAX_ZIP_EXTRA} comparable ZIPs, got {len(comp)}")
        else:
            picks[z], reasons[z] = comp, str(r.get("reason", ""))[:160]
    missing = [z for z in allowed if z not in picks and not any(e.startswith(f"{z}:") for e in errors)]
    if missing:
        errors.append(f"no ranking for target(s) {missing}")
    return picks, reasons, errors


def rule_based_neighbours(user: str) -> str:
    """Deterministic stand-in with the same prompt -> JSON contract: keeps the geographic order."""
    ev = json.loads(user.split(MARK, 1)[1].split("\n", 1)[0])
    return json.dumps({"rankings": [{"zip_code": t["zip_code"], "reason": "nearest first",
                                     "comparable": [c["zip_code"] for c in t["candidates"][:MAX_ZIP_EXTRA]]}
                                    for t in ev["targets"]]})


def rank_neighbours(llm, c: Criteria, snap: Snapshot, min_listings: int = MIN_LISTINGS,
                    max_attempts: int = 2) -> tuple[dict[str, list[str]], list[str]]:
    """LLM ranks comparable neighbours for the thin ZIPs, code validates. -> ({zip: full neighbour order}, trace)."""
    from ..parser_agent import _extract_json as extract_json
    targets = thin_zips(c, snap, min_listings) if llm is not None else []
    if not targets:
        return {}, []
    picks, reasons, notes = {}, {}, []
    for i in range(0, len(targets), RANK_BATCH):
        batch = targets[i:i + RANK_BATCH]
        allowed = {z: candidates(snap, z) for z in batch}
        user, feedback = build_neighbour_prompt(batch, snap), ""
        for attempt in range(1, max_attempts + 1):
            try:
                obj = extract_json(llm(SYSTEM_PROMPT, user + feedback))
            except Exception as ex:
                notes.append(f"attempt {attempt}: unusable reply ({ex})")
                feedback = f"\n\nYour previous reply was not valid JSON ({ex}). Reply with ONLY the JSON object."
                continue
            got, why, errors = validate_rankings(obj, allowed)
            picks.update(got)
            reasons.update(why)
            allowed = {z: v for z, v in allowed.items() if z not in got}      # retry only what is still missing
            if not errors or not allowed:
                break
            notes.append(f"attempt {attempt}: rejected: {'; '.join(errors[:3])}")
            feedback = (f"\n\nYour previous reply was rejected: {'; '.join(errors[:3])}. Fix it and reply with ONLY "
                        f"the JSON object, with rankings for {sorted(allowed)}.")

    order, changed = {}, []
    for z, comp in picks.items():
        geo = snap.neighbours(z, len(snap.zips))
        order[z] = comp + [o for o in geo if o not in comp]
        if order[z][:MAX_ZIP_EXTRA] != geo[:MAX_ZIP_EXTRA]:
            changed.append(f"scanner:   {z} pools {', '.join(order[z][:MAX_ZIP_EXTRA])} "
                           f"(geographic: {', '.join(geo[:MAX_ZIP_EXTRA])}): {reasons.get(z, '')}")
    fallback = [z for z in targets if z not in picks]
    trace = [f"scanner:   neighbours ranked by {getattr(llm, 'name', 'llm')} for {len(picks)}/{len(targets)} thin ZIP(s), "
             f"{len(changed)} differ from the geographic order"
             + (f"; geographic fallback for {fallback}" if fallback else "")]
    return order, trace + changed[:10] + [f"scanner:   {n}" for n in notes]


def make_scanner(snap: Snapshot, min_listings: int = MIN_LISTINGS, llm=None):
    """LangGraph node factory: reads state['criteria'] and state['scope'], writes state['scan']. The neighbour order is
    ranked on the first call only and kept in state['neighbour_order'], so Pooling's rescans reuse it."""
    def scanner(state):
        order, extra = state.get("neighbour_order"), []
        if order is None:
            order, extra = rank_neighbours(llm, state["criteria"], snap, min_listings)
        res = scan(state["criteria"], state.get("scope") or Scope(), snap, min_listings, order=order)
        return {"scan": res, "neighbour_order": order, "trace": [f"scanner: {res.summary()}"] + extra}
    return scanner