# Metamorphic relation catalog

Every relation is a statement about the **output contract**: one row per month of the 12-month window over all requested ZIPs (own listings only), with the columns `price`, `ppsf`, `n`, `mom_price_pct` (median list price, median price per sq ft, active-listing count, month-over-month price %).

An MR is a hypothesis: a violation is a symptom that needs a cause (agent/aggregation fault, data artifact, ill-conceived MR, or harness bug), not automatically a bug.

## MR1: Reorder listings (Invariance)

- **Relation:** Shuffling the rows of the listing file leaves the whole contract (overall and per ZIP-month) unchanged.
- **Subdomain:** Every base input.
- **Tolerance:** Exact: relative 1e-9 on every contract column.
- **Fault exposed if violated:** Order dependence: a first/last-row choice, an unstable sort or an unkeyed groupby in the Scanner or Statistician.
- **Justification:** Row order carries no meaning in a listing feed, so any change is a defect, not a modelling choice. Baseline sanity: if this fails, no other result is interpretable.

## MR2: Sort listings by price (adversarial order) (Invariance)

- **Relation:** Sorting the rows ascending by price inside every ZIP-month leaves the whole contract unchanged.
- **Subdomain:** Every base input.
- **Tolerance:** Exact: relative 1e-9 on every contract column.
- **Fault exposed if violated:** Position-dependent logic that a random shuffle rarely triggers: head/tail slicing, 'first N' caps or id/position tie-breaks that matter only when the cheapest or dearest listings come first.
- **Justification:** A sorted feed is just another valid order; a random shuffle and an adversarial order exercise different failure modes.

## MR3: Add a 10x-priced fake month outside the window (Invariance)

- **Relation:** Adding listings in a month far outside the query window (and beyond Pooling's padding) leaves the whole contract unchanged.
- **Subdomain:** Every base input.
- **Tolerance:** Exact: relative 1e-9 on every contract column.
- **Fault exposed if violated:** Out-of-window data leaking into the answer: the Critic's ZipProfile (log-price median/MAD behind the outlier z-scores) is computed from the WHOLE snapshot, so out-of-window rows can change which listings count as suspects.
- **Justification:** The query asks about 12 months; listings of other months are not part of the question. If the answer depends on them, the system's scope is wider than its contract.

## MR4: Multiply each ZIP's listings by a random N (1-10) (Invariance)

- **Relation:** With every ZIP's listings present N_zip times (fresh ids, N_zip random per ZIP), each requested ZIP's median price, $/sqft and month-over-month % are unchanged and its monthly own count is exactly N_zip times larger.
- **Subdomain:** Every base input (violations are expected where buckets cross the coverage threshold: n changes which buckets are thin).
- **Tolerance:** Exact: relative 1e-9 on price, ppsf, mom_price_pct per ZIP-month; n_own must be exactly N_zip x.
- **Fault exposed if violated:** Estimators that are not scale-free in sample size, coverage logic that reacts to the count, or id-keyed exclusions that do not follow a listing's copies.
- **Justification:** Replicating a ZIP's sample leaves its empirical distribution unchanged, so any distribution statistic is unchanged. The coverage threshold depends on n by design, so this MR is partly ill-conceived on thin buckets.

## MR5: Add $X to every price in one month (Monotonic)

- **Relation:** Adding $50,000 to every price in one month raises that month's median by ~$50,000 (within 1% of X) and no month's median decreases.
- **Subdomain:** Base inputs without a price filter (a price band would move listings in or out).
- **Tolerance:** Shift within 1% of X; no decrease beyond 1e-6.
- **Fault exposed if violated:** Statistics that react to the injected shift (a jump triggers the robust recompute, which replaces the median by a trimmed mean), or month bucketing errors.
- **Justification:** The median is translation-equivariant, so a uniform shift of a month's prices shifts its median by the same amount. The system's own jump logic is expected to interfere.

## MR6: Remove the dearest listing of each ZIP (Monotonic)

- **Relation:** Deleting the single most expensive in-scope listing of every requested ZIP raises no ZIP-month median.
- **Subdomain:** Every base input.
- **Tolerance:** No increase beyond 1e-6 in any ZIP-month with a median on both sides.
- **Fault exposed if violated:** Buckets pushed across the coverage threshold (the removal makes a bucket thin, Pooling then borrows neighbours that raise its median), or any estimator that is not monotone in its sample.
- **Justification:** Removing the maximum of a sample can never raise its median, so only the pipeline's reaction (pooling, robust recompute, exclusions) can break it.

## MR7: Widen the price band (Monotonic)

- **Relation:** Lowering price_min by 25% and/or raising price_max by 25% (a superset of listings) never lowers the listing count of any month.
- **Subdomain:** Base inputs that have a price_min or price_max.
- **Tolerance:** Exact: follow-up n >= source n for every month.
- **Fault exposed if violated:** Filters applied wrongly (inclusive/exclusive bounds), or exclusions that remove more listings after widening.
- **Justification:** A superset of listings cannot have fewer members; only an agent that drops listings differently can break it.

## MR8: Restrict to a sub-region (Monotonic)

- **Relation:** Keeping a strict subset of the ZIPs never gives any month more listings than the full set.
- **Subdomain:** Base inputs with at least two ZIPs.
- **Tolerance:** Exact: follow-up n <= source n for every month.
- **Fault exposed if violated:** ZIP scoping errors, or pooled listings leaking into the 'own' count.
- **Justification:** Own listings of fewer ZIPs are a subset of own listings of more ZIPs. Tests ZIP isolation from the count side (MR9 tests it from the price side).

## MR9: Double all prices in one ZIP (Directional)

- **Relation:** Multiplying every price in ONE ZIP by 2 doubles that ZIP's median price and $/sqft in every month and leaves its counts and % changes, and every other ZIP's rows, unchanged.
- **Subdomain:** Base inputs without a price filter (a price band is not scaled).
- **Tolerance:** Exact: relative 1e-9 on every ZIP-month row.
- **Fault exposed if violated:** Cross-ZIP leakage: Pooling borrows listings from neighbouring ZIPs, so a changed ZIP changes the medians of its thin neighbours (and the doubled ZIP's own pooled buckets stop scaling cleanly).
- **Justification:** A ZIP's statistics describe that ZIP's listings. MR9 tests a local scale change; MR12 a global one: if MR12 holds and MR9 fails, the cause is cross-ZIP pooling, not a scale-sensitive threshold.

## MR10: Inject one 10x outlier (Directional)

- **Relation:** Multiplying the price of one counted listing by 10 moves no month's overall median, and no month of its ZIP, by 15% or more; and whenever the Diagnoser examines that ZIP, the planted listing is among its suspects.
- **Subdomain:** Base inputs without a price filter (the 10x price would leave a band).
- **Tolerance:** Relative change < 15% (overall and ZIP-month); suspect list membership.
- **Fault exposed if violated:** Fragile estimator or loop reaction: one listing flips a ZIP to the robust trimmed mean (or triggers exclusions), moving the whole series; or the outlier screen does not surface a 10x listing the Diagnoser should judge.
- **Justification:** A single extreme value is exactly what the median, the Critic's screen and the Diagnoser exist to absorb.

## MR11: Replace a month with a copy of the previous month (Directional)

- **Relation:** If one month's listings are replaced by a copy of the previous month's, the two months have equal median price, median $/sqft and count, and the month-over-month change into the replaced month is 0%.
- **Subdomain:** Every base input.
- **Tolerance:** Exact: relative 1e-9; mom_price_pct within 1e-6 of 0.
- **Fault exposed if violated:** Identical months treated differently: thin-bucket pooling or robust recompute that depends on neighbouring months, id-keyed exclusions that hit one copy but not the other.
- **Justification:** Two months with the same listings have the same distribution, so any per-month statistic must agree.

## MR12: Scale all prices by k > 1 (Directional)

- **Relation:** Multiplying every price by k (k in {1.5, 2, 3}) scales median price and $/sqft by k (overall and per ZIP-month) and leaves the month-over-month % and the counts unchanged.
- **Subdomain:** Base inputs without a price filter (a price band is not scaled).
- **Tolerance:** Relative 1e-9 on every column after applying k.
- **Fault exposed if violated:** Absolute (non-relative) thresholds or rounding in the Critic, Diagnoser evidence or Statistician.
- **Justification:** All of the system's tests are meant to be relative (percentage jumps, log-price z-scores), so a currency rescaling must not change a decision.
