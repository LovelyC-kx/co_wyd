"""
Aggregation-scheme tags + plot colours for the v1 cloud-removal package.

Intentionally duplicated from ../constants.py so this folder is a
self-contained unit: the only imports from outside cloud_removal_v1
are `torch`, `numpy`, `spikingjelly`, `PIL`, `matplotlib`.  No fragile
cross-package relative imports.
"""

# Plot palette (matches the original FLSNN paper's Fig 5 styling)
color_list  = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#000000", "#77AC30"]
marker_list = ["o", "s", "d", "*", "|", "x"]

# Inter-plane aggregation scheme tags
GOSSIP    = "Gossip_Averaging"
RELAYSUM  = "Relaysum_Aggregation"
ALLREDUCE = "AllReduce_Aggregation"
SCHEMES   = (RELAYSUM, GOSSIP, ALLREDUCE)

# Legend labels for plots (human-readable)
SCHEME_LABEL = {
    RELAYSUM:  "RelaySum (Proposed)",
    GOSSIP:    "Gossip",
    ALLREDUCE: "All-Reduce",
}
