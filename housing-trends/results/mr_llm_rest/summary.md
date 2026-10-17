| MR | Family | Relation | Base inputs | Seeds | Violating base inputs | Rate (95% Wilson CI) | Violating runs / runs | Skipped | Errors |
|---|---|---|---|---|---|---|---|---|---|
| MR1 | Invariance | Sort listings by price (adversarial order) | 30 | 101 202 303 | 0 | 0.0% (0.0-11.4) | 0 / 90 | 0 | 0 |
| MR3 | Monotonic | Add $X to every price in one month | 30 | 101 202 303 | 19 | 63.3% (45.5-78.1) | 40 / 90 | 0 | 0 |
| MR5 | Monotonic | Widen the price band | 30 | 101 202 303 | 0 | 0.0% (0.0-11.4) | 0 / 90 | 0 | 0 |
| MR6 | Monotonic | Restrict to a sub-region | 30 | 101 202 303 | 1 | 3.3% (0.6-16.7) | 1 / 89 | 1 | 0 |
| MR8 | Directional | Inject one 10x outlier | 30 | 101 202 303 | 9 | 30.0% (16.7-47.9) | 13 / 90 | 0 | 0 |
| MR9 | Directional | Replace a month with a copy of the previous month | 30 | 101 202 303 | 4 | 13.3% (5.3-29.7) | 6 / 90 | 0 | 0 |
| MR10 | Directional | Scale all prices by k > 1 | 30 | 101 202 303 | 22 | 73.3% (55.6-85.8) | 64 / 90 | 0 | 0 |
