"""
cloud_removal_v2 — Workshop-tier extension of cloud_removal_v1.

Path A delivery:
  * Multi-source dataset (CUHK-CR1 + CUHK-CR2) with source labels.
  * Synchronized geometric augmentation on paired (cloudy, clear) images.
  * Dirichlet(α) partition over source labels → realistic feature-shift
    non-IID (thin vs thick clouds).
  * Sequential runner that sweeps BN-strategy × aggregation-scheme
    (6 runs: FedAvg × {RelaySum, Gossip, All-Reduce}
     plus    FedBN × same three).
  * Qualitative comparison grid (cloudy / 6 model restores / clear).

This package reuses cloud_removal_v1's models, aggregation, evaluation,
constellation, and constants packages verbatim; only data pipeline,
config defaults, runner, plotting and visualisation are new.

Imports are lazy: importing `cloud_removal_v2` alone does NOT trigger
spikingjelly / torch-CUDA initialisation, so tests/run_all can be run
on a minimal environment.

Recommended import style (matches v1 convention):

    from cloud_removal_v2.dataset import (
        MultiSourceCloudDataset,
        AugmentedPairedCloudDataset,
        dirichlet_source_partition,
        build_plane_satellite_partitions_v2,
    )
    from cloud_removal_v2.config import V2A_DEFAULTS, build_v2a_args, parse_v2a_cli
    from cloud_removal_v1.constellation import CloudRemovalConstellation
    from cloud_removal_v1.constants import GOSSIP, RELAYSUM, ALLREDUCE
"""

__version__ = "2.0.0a"
