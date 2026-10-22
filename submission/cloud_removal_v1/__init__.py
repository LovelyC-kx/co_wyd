"""
cloud_removal_v1
================
Self-contained v1 package for decentralized cloud removal with VLIFNet.

Public API
----------
Submodules are intentionally **not** eagerly imported at package load —
importing e.g. `cloud_removal_v1.aggregation` only pulls in numpy+torch,
whereas `cloud_removal_v1.models` / `.task` / `.constellation` /
`.evaluation` also need `spikingjelly`.  This lets pure tensor-math unit
tests (cloud_removal_v1.tests.run_all) run on a machine without
spikingjelly installed.

Recommended import style:

    from cloud_removal_v1.models import build_vlifnet
    from cloud_removal_v1.dataset import PairedCloudDataset
    from cloud_removal_v1.task import CloudRemovalSNNTask
    from cloud_removal_v1.constellation import CloudRemovalConstellation
    from cloud_removal_v1.evaluation import evaluate_per_plane
    from cloud_removal_v1.config import build_v1_args, parse_v1_cli
    from cloud_removal_v1.aggregation import (
        is_bn_key, average_state_dicts, apply_aggregated,
    )
    from cloud_removal_v1.constants import (
        GOSSIP, RELAYSUM, ALLREDUCE, SCHEMES, SCHEME_LABEL,
    )

Entry points:
    python -m cloud_removal_v1.run_smoke    [CLI flags]
    python -m cloud_removal_v1.plot_results --run_name <run>
    python -m cloud_removal_v1.tests.run_all
"""

__version__ = "1.0.0"
