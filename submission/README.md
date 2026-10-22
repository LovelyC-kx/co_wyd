# ORBITALIF — Code Submission

Source code for the paper *"ORBITALIF: An Efficient Spiking Federated
Learning Framework for Onboard Cloud Removal"*. Self-contained: the two
Python packages below plus the listed dependencies are everything needed
to reproduce the paper's experiments (the CUHK-CR dataset must be
obtained separately, see §3).

## 1. Layout

```
cloud_removal_v1/            centralized training + FL primitives + models
  models/vlifnet.py            ORBITALIF backbone (SNN / ANN variants), AGFM
  models/fsta_module.py        SHAM (temporal + spectral attention), FreqMLP
  models/plain_unet.py         plain U-Net baseline (ORBITUNET)
  models/_sj_compat.py         spikingjelly 0.0.0.0.14 compatibility shims
  train_centralized.py         Table I main results + ablations (see file header)
  task.py                      per-satellite trainer (loss, AdamW, LR schedule)
  aggregation.py               state-dict averaging primitives (+ self-tests)
  constellation.py             5x10 constellation orchestrator (ring + gossip)
  dataset.py / evaluation.py   data loading, PSNR/SSIM evaluation
  config.py / run_smoke.py     federated run configuration + entry point
  plot_paper_figs.py           paper figure generation
  tests/run_all.py             pure-CPU self-tests (no dataset needed)

cloud_removal_v2/            federated experiments + energy accounting
  dataset.py                   CR1∪CR2 multi-source Dirichlet(α) partition
  config.py / run_smoke.py     federated experiment configuration + entry point
  energy_estimation.py         per-layer MAC/SOP hooks (Conv2d/Conv3d/Linear)
  energy_audit_offline.py      dataset-free energy audit (reproduces Fig 3b)
  tests/run_all.py             pure-CPU self-tests (no dataset needed)

energy_audit.json            output of energy_audit_offline (reference numbers)
requirements.txt             pinned dependencies
```

`cloud_removal_v2` imports models from `cloud_removal_v1`; keep the two
directories side by side and run all commands from this directory.

## 2. Installation

```bash
pip install -r requirements.txt
```

`spikingjelly` MUST be `0.0.0.0.14` — `models/_sj_compat.py` patches two
known bugs of exactly that release. Python ≥ 3.9, PyTorch ≥ 2.0 (CPU or
CUDA). `tensorboard` is optional (training logs degrade gracefully
without it).

## 3. Dataset

CUHK-CR1 (thin cloud) and CUHK-CR2 (thick cloud) are the public
benchmark of Sui et al., TGRS 2024 (arXiv:2401.15105). Accepted
directory layouts (auto-detected by `cloud_removal_v1/dataset.py`):

```
<root>/train/input/ + <root>/train/target/   and same for test/   (preferred)
<root>/cloudy/ + <root>/clear/                                    (flat variant)
```

Pass the root via `--data_root`.

## 4. Verify the installation (no dataset, no GPU needed)

```bash
python -m cloud_removal_v1.tests.run_all        # aggregation + partition self-tests
python -m cloud_removal_v2.tests.run_all        # Dirichlet partition + augment self-tests
python -m cloud_removal_v1.aggregation          # 4 aggregation self-tests
python -m cloud_removal_v2.energy_audit_offline # parameter counts + MAC audit
```

Expected from the last command: VLIFNet = 2,302,901 params (2.30 M),
PlainUNet = 1,311,603 params, Conv2d MACs = 4.511 G per 64x64 inference.

## 5. Reproducing the paper's numbers

| Paper item | Command |
|---|---|
| Table I, ORBITALIF 25.374 dB (CR1, 400 ep) | `python -m cloud_removal_v1.train_centralized --data_root <CR1> --dataset_name CR1 --backbone vlif --num_epoch 400 --run_name A1_vlif_cr1` |
| Table I, w/o SHAM ablation | same + `--ablation no_fsta --num_epoch 300` |
| Table I, w/o AGFM ablation | same + `--ablation no_agfm --num_epoch 300` |
| Binary-spike ablation | same + `--ablation binary_spike --num_epoch 300` |
| ORBITUNET baseline | same + `--backbone plain_ann` |
| Federated (Fig 3a): ORBITALIF / ORBITANN / ORBITUNET, 5x10 Walker-Star, Dirichlet(0.1), 200 rounds | `python -m cloud_removal_v2.run_smoke --num_epoch 200 ...` (see `cloud_removal_v2/config.py` for the full flag set: `partition_mode=dirichlet_source`, `partition_alpha=0.1`, `bn_local` for FedBN, `--backbone snn|ann|plain`) |
| Energy (Fig 3b), full measurement with real spike rates | `python -m cloud_removal_v2.energy_estimation --ckpt <A1 ckpt> --data_root <CR1> --split test --n_samples 32 --patch_size 64 --out_dir Outputs/energy_A1` |
| Energy, dataset-free audit (MACs + parameter counts) | `python -m cloud_removal_v2.energy_audit_offline --out energy_audit.json` |

Full CLI documentation lives in each entry point's module docstring
(`train_centralized.py` header documents every ablation mode).

## 6. Note on the energy meter (post-paper correction)

The energy meter originally hooked only `Conv2d`/`Linear` layers. This
submission's `energy_estimation.py` additionally hooks `Conv3d` (the
backbone's temporal-compression layers, 28.3 MMAC per inference, dense
non-event-driven ops) and reports them separately:

* legacy fields (`energy_snn_lower_pj`, ...) keep the paper's original
  semantics and reproduce the published figures exactly
  (0.287 mJ, 20.75→20.88 mJ ANN, 72.3x);
* the new field `energy_snn_lower_dense3d_pj` charges Conv3d at the full
  4.6 pJ/MAC rate inside the SNN estimate, giving the corrected bound
  0.417 mJ (50.1x vs the T=4 ANN twin; conservative upper bound
  17.27 mJ / 1.21x unchanged).

`energy_audit.json` in this directory contains both sets of numbers as
produced by `energy_audit_offline.py`.
