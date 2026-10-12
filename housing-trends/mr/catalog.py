"""Write the MR catalog (relation, subdomain, tolerance, fault exposed, justification) from relations.MRS, so the
document can never drift from the code that is actually executed."""
from __future__ import annotations

from .contract import CONTRACT_COLUMNS
from .relations import MRS


def catalog_markdown() -> str:
    out = ["# Metamorphic relation catalog\n",
           "Every relation is a statement about the **output contract**: one row per month of the 12-month window over all "
           f"requested ZIPs (own listings only), with the columns `{'`, `'.join(CONTRACT_COLUMNS)}` "
           "(median list price, median price per sq ft, active-listing count, month-over-month price %).\n",
           "An MR is a hypothesis: a violation is a symptom that needs a cause (agent/aggregation fault, data artifact, "
           "ill-conceived MR, or harness bug), not automatically a bug.\n"]
    for m in MRS:
        out.append(f"## {m.id}: {m.name} ({m.family})\n")
        out += [f"- **Relation:** {m.relation}", f"- **Subdomain:** {m.subdomain}", f"- **Tolerance:** {m.tolerance}",
                f"- **Fault exposed if violated:** {m.fault}", f"- **Justification:** {m.justification}\n"]
    return "\n".join(out)
