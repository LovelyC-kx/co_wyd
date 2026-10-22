# cloud_removal_v2 — Path-A (workshop-tier) extension of v1

## Scope

v2-A is the first deliverable on the 3-path v2 roadmap.  It upgrades the
v1 baseline in three targeted ways, leaving everything else (constellation,
RelaySum algorithm, VLIFNet backbone, compat shims) unchanged:

1. **Multi-source dataset** — combines CUHK-CR1 (thin) and CUHK-CR2 (thick)
   into one training pool with per-sample source labels.
2. **Synchronized geometric augmentation** — horizontal / vertical flip
   and ±90° rotation applied IDENTICALLY to (cloudy, clear).  No colour
   jitter (remote-sensing radiance has physical meaning).
3. **Dirichlet-over-source non-IID partition** — each satellite receives
   a cloud-type mixture drawn from Dirichlet(α=0.1).  α controls
   non-IID-ness; α=0.1 is aggressive feature shift.

On top of this, the runner sweeps **2 BN modes × 3 aggregation schemes =
6 sequential runs** in a single script invocation, and the visualiser
produces a qualitative grid comparing all six models on a common test
patch set.

## Expected results vs v1

| Expectation | Rationale |
|---|---|
| RelaySum > Gossip by ≥ 0.5 dB PSNR | Thm 2 ζ² term becomes non-zero under Dirichlet(0.1); RelaySum's relay buffers benefit from cross-client information flow |
| FedBN > FedAvg by ≥ 0.3 dB on any scheme | TDBN statistics accumulate feature-shift-specific information that averaging across clients dilutes; cf. FedBN ICLR'21 |
| Final PSNR ≥ 22 dB, SSIM ≥ 0.70 | v1 baseline at 10 epoch reached 20.8 dB; 30 epoch + augment should yield ≥ 1.5 dB gain |
| Qualitative viz: cloud visibly removed | Eye-check; bare minimum for a cloud-removal paper |

If any expectation fails by > 50 %, the Path-A run should be diagnosed
before moving to Path B or C.

## Layout

```
cloud_removal_v2/
├── __init__.py                   lazy package
├── README.md                     this file
├── config.py                     V2A_DEFAULTS + build_v2a_args + parse_v2a_cli
├── dataset.py                    MultiSourceCloudDataset, AugmentedPairedCloudDataset,
                                    dirichlet_source_partition,
                                    build_plane_satellite_partitions_v2
├── task.py                       re-exports v1 CloudRemovalSNNTask + losses
├── run_smoke.py                  6-run sweep + checkpoints + summary
├── plot_results.py               6-curve comparison PDFs
├── visualize.py                  qualitative (cloudy | 6 restores | clear) grid
├── docs/
│   └── v2a_setup.md              run book + troubleshooting
└── tests/
    └── run_all.py                Dirichlet + augment unit tests
```

## Quickstart on AutoDL (4090)

```bash
# 1. Pull latest
cd /root/autodl-tmp/shiyaunmingFLSNN-main/Decentralized-Satellite-FL-dev-main
git pull origin <branch>

# 2. Self-tests (pure Python / torch CPU; no GPU or spikingjelly needed)
python -m cloud_removal_v2.tests.run_all

# 3. Dataset probe
python -m cloud_removal_v2.dataset /root/autodl-tmp/C-CUHK

# 4. Sanity smoke (3 plane × 3 sat × 3 epoch ≈ 20 min)
python -m cloud_removal_v2.run_smoke \
    --data_root /root/autodl-tmp/C-CUHK \
    --num_epoch 3 --num_planes 3 --sats_per_plane 3 \
    --run_name v2a_dryrun

# 5. Full sweep (30 ep × 50 sat × 6 runs ≈ 15 h on 4090) — USE tmux!
tmux new -s v2a
python -m cloud_removal_v2.run_smoke --data_root /root/autodl-tmp/C-CUHK 2>&1 \
    | tee Outputs_v2/v2a.log
# Ctrl-B D to detach.  tmux attach -t v2a to resume.

# 6. Plots + qualitative
python -m cloud_removal_v2.plot_results --run_name v2a
python -m cloud_removal_v2.visualize   --run_name v2a --n_samples 6
```

## Cell matrix produced by a full sweep

```
                     RelaySum       Gossip         All-Reduce
                  +-------------+---------------+---------------+
    FedAvg        | R1 (v1-like) | R2 (v1-like) | R3 (v1-like) |
                  +-------------+---------------+---------------+
    FedBN (NEW)   | R4          | R5            | R6            |
                  +-------------+---------------+---------------+
```

Each cell produces one `.npz`, one `.pt` (plane 0 for FedAvg; 5 planes
for FedBN), and contributes one curve to each of the 3 comparison PDFs.

## See also

* `docs/v2a_setup.md` — environment / data / troubleshooting.
* `../cloud_removal_v1/docs/paper_section_6_draft.md` — §VI scaffold
  where v2-A results fill in VI-B (non-IID) and VI-C (FedBN).
