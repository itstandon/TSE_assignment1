"""Metamorphic-testing harness for the housing-trends agent graph.

contract.py    the output contract: exactly which numbers "the 12-month trend" consists of
compare.py     tolerance-aware comparison helpers used by the relations
stats.py       violation rate and Wilson 95% confidence interval
inputs.py      reproducible generator of base inputs (criteria sets) for the snapshot
transforms.py  the input transformations (reorder, duplicate, scale, shift window, ...)
relations.py   the metamorphic relations: transform + expected relation + subdomain + tolerance + fault
harness.py     source run -> transform -> follow-up run -> check, over base inputs and seeds
catalog.py     writes the MR catalog (docs/mr_catalog.md) from the relations
"""
