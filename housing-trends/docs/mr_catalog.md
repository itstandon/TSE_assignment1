# Metamorphic relation catalog

Every relation is a statement about the **output contract**: one row per month of the 12-month window over all requested ZIPs (own listings only), with the columns `price`, `ppsf`, `n`, `mom_price_pct` (median list price, median price per sq ft, active-listing count, month-over-month price %).

An MR is a hypothesis: a violation is a symptom that needs a cause (agent/aggregation fault, data artifact, ill-conceived MR, or harness bug), not automatically a bug.

## MR1: Reorder listings (Invariance)

- **Relation:** Shuffling the rows of the listing file leaves the whole contract unchanged.
- **Subdomain:** Every base input.
- **Tolerance:** Exact: relative 1e-9 on all four contract columns.
- **Fault exposed if violated:** Order dependence: a first/last-row choice, an unstable sort or an unkeyed groupby in the Scanner or Statistician.
- **Justification:** Row order carries no meaning in a listing feed, so any change is a defect, not a modelling choice.

## MR2: Relabel listing ids (Invariance)

- **Relation:** Giving the same listings different ids (a permutation of the id set) leaves the contract unchanged.
- **Subdomain:** Every base input.
- **Tolerance:** Exact: relative 1e-9 on all four contract columns.
- **Fault exposed if violated:** Logic that depends on listing_id values: id-based tie-breaking, caps or sorts (the Critic's suspect cap and the Diagnoser's exclusions are ordered by id), or ids used as positions.
- **Justification:** A listing's id is a label; behaviour must not depend on which label it carries.

## MR3: Duplicate every listing (Invariance)

- **Relation:** With every listing present twice, medians and $/sqft and the month-over-month % are unchanged and the monthly listing count exactly doubles.
- **Subdomain:** Every base input (violations are expected where thin buckets exist: more listings change which buckets are 'thin').
- **Tolerance:** Exact: relative 1e-9 on price, ppsf and mom_price_pct; n must be exactly 2x.
- **Fault exposed if violated:** Estimators that are not scale-free in sample size (floor-based trimming, n-dependent weights), or coverage logic that reacts to the count.
- **Justification:** Duplicating a sample leaves its empirical distribution unchanged, so any distribution statistic is unchanged. The system's coverage threshold depends on n by design, so this MR may be ill-conceived on thin buckets.

## MR4: Unit change: USD to USD thousands (Invariance)

- **Relation:** Expressing prices (and the price filter) in thousands scales price and $/sqft by 1/1000 and leaves counts and month-over-month % unchanged.
- **Subdomain:** Every base input (price filters are whole thousands by construction).
- **Tolerance:** Relative 1e-9 on all four columns after applying the factor.
- **Fault exposed if violated:** Hard-coded dollar magnitudes (rounding, absolute thresholds, formatting) in the agents.
- **Justification:** A display unit must not change rankings or percentages.

## MR5: Widen the price band (Monotonic)

- **Relation:** Lowering price_min by 25% and/or raising price_max by 25% (a superset of listings) never lowers the listing count of any month.
- **Subdomain:** Base inputs that have a price_min or price_max.
- **Tolerance:** Exact: follow-up n >= source n for every month.
- **Fault exposed if violated:** Filters applied wrongly (inclusive/exclusive bounds), or exclusions that remove more listings after widening.
- **Justification:** A superset of listings cannot have fewer members; only an agent that drops listings differently can break it.

## MR6: Restrict to a sub-region (Monotonic)

- **Relation:** Keeping a strict subset of the ZIPs never gives any month more listings than the full set.
- **Subdomain:** Base inputs with at least two ZIPs.
- **Tolerance:** Exact: follow-up n <= source n for every month.
- **Fault exposed if violated:** ZIP scoping errors, or pooled listings leaking into the 'own' count.
- **Justification:** Own listings of fewer ZIPs are a subset of own listings of more ZIPs.

## MR7: Add $X to one month (Monotonic)

- **Relation:** Adding $50,000 to every price in one month raises that month's median by ~$50,000 (within 1% of X) and no month's median decreases.
- **Subdomain:** Base inputs without a price filter (a price band would move listings in or out).
- **Tolerance:** Shift within 1% of X; no decrease beyond 1e-6.
- **Fault exposed if violated:** Statistics that react to the injected shift (outlier handling excluding or trimming the shifted month), or month bucketing errors.
- **Justification:** The median is translation-equivariant, so a uniform shift of a month's prices shifts its median by the same amount.

## MR8: Shift the window forward one month (Directional)

- **Relation:** Moving the 12-month window forward by one month leaves the 11 overlapping months unchanged (price, $/sqft, n; month-over-month % for months with a prior in both runs).
- **Subdomain:** Base inputs whose window can move forward inside the snapshot.
- **Tolerance:** Relative 1e-9 on the overlapping months.
- **Fault exposed if violated:** Results that depend on the window position: loop decisions (robust recompute, exclusions) driven by months outside the overlap, or edge effects in pooling.
- **Justification:** A month's statistics describe that month's listings, not which other months were requested.

## MR9: Scale all prices by k > 1 (Directional)

- **Relation:** Multiplying every price by k (k in {1.5, 2, 3}) scales median price and $/sqft by k and leaves the month-over-month % and the counts unchanged.
- **Subdomain:** Base inputs without a price filter (a price band is not scaled).
- **Tolerance:** Relative 1e-9 on all four columns after applying k.
- **Fault exposed if violated:** Absolute (non-relative) thresholds or rounding in the Critic, Diagnoser evidence or Statistician.
- **Justification:** All of the system's tests are meant to be relative (percentage jumps, log-price z-scores), so a currency rescaling must not change a decision.

## MR10: Raise one ZIP's prices (Directional)

- **Relation:** Multiplying every price in one ZIP by 1.5 never lowers any month's overall median price.
- **Subdomain:** Base inputs without a price filter.
- **Tolerance:** No decrease beyond 1e-6 in any month.
- **Fault exposed if violated:** Loops that react to the injected change (a new 'jump' triggering exclusions that remove listings elsewhere), or ZIP scoping errors.
- **Justification:** Raising some prices can only move a median up or leave it unchanged.
