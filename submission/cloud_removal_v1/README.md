# cloud_removal_v1 — self-contained v1 package

Decentralized federated learning with VLIFNet for optical cloud removal
on CUHK-CR1 / RICE1.  Reproduces the three-scheme comparison from the
original FLSNN paper (RelaySum vs Gossip vs AllReduce) but for image
regression.

Everything this package needs is inside this folder.  External deps are
limited to `torch`, `spikingjelly==0.0.0.0.14`, `PIL`, `numpy`,
`matplotlib`.

## Layout

```
cloud_removal_v1/
├── __init__.py                # public API re-exports
├── constants.py               # scheme tags + plot colours
├── config.py                  # V1_DEFAULTS + build_v1_args + parse_v1_cli
├── aggregation.py             # state_dict math (tensor-aware + FedBN hook)
├── dataset.py                 # PairedCloudDataset + partitioners
├── task.py                    # CloudRemovalSNNTask (per-satellite)
├── constellation.py           # CloudRemovalConstellation (3 schemes)
├── evaluation.py              # PSNR/SSIM (center_patch | fullimage | sliding)
├── run_smoke.py               # top-level entry point
├── plot_results.py            # figure generator
├── models/
│   ├── __init__.py
│   ├── vlifnet.py             # VLIFNet U-Net (vendored)
│   └── fsta_module.py         # FSTA + FreMLPBlock (vendored)
├── tests/
│   └── run_all.py             # pure-Python self-tests
└── docs/
    ├── setup.md               # env + data + run + troubleshoot
    └── model_comparison.md    # FLSNN vs ESDNet vs VLIFNet
```

## Quickstart

```bash
cd <project_root>/Decentralized-Satellite-FL-dev-main

# 1. Env
conda create -n flsnn_cr python=3.9 -y && conda activate flsnn_cr
conda install pytorch==2.1.* torchvision==0.16.* pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
pip install spikingjelly==0.0.0.0.14 thop tensorboard tqdm Pillow \
            scikit-image numpy matplotlib

# 2. Self-tests (no GPU needed)
python -m cloud_removal_v1.tests.run_all

# 3. Data — either symlink your CUHK-CR1 or pass --data_root
python -m cloud_removal_v1.dataset /abs/path/to/CUHK-CR1

# 4. Smoke run (3 schemes × 10 epochs × 50 satellites ≈ 2–4 h on V100)
python -m cloud_removal_v1.run_smoke --data_root /abs/path/to/CUHK-CR1

# 5. Plots
python -m cloud_removal_v1.plot_results --run_name v1_smoke
```

Detailed instructions: `docs/setup.md`.
Model background + paper comparison: `docs/model_comparison.md`.
