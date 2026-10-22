"""
Decentralized constellation orchestrator (v1).

Holds `num_planes × sats_per_plane` CloudRemovalSNNTask instances and
drives the full FLSNN training loop:

    for each global round:
        for _ in intra_plane_iters:
            for each satellite:   local_training()
            for each plane:       intra-plane equal-weight average + broadcast
        inter-plane aggregation (RELAYSUM / GOSSIP / ALLREDUCE)
        broadcast new plane states to every satellite
        advance global round counter

Aggregation primitives live in aggregation.py so the bn_local (FedBN)
knob is controlled from a single place.  Relay buffers live on the
same device as the per-satellite weights to avoid CPU↔GPU mismatches.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import GOSSIP, RELAYSUM, ALLREDUCE
from .task import CloudRemovalSNNTask
from .aggregation import (
    is_bn_key,
    average_state_dicts,
    state_div,
    zeros_like_state,
)


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def _clone_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep-clone tensor entries; pass non-tensor entries through."""
    return {k: (v.detach().clone() if _is_tensor(v) else v)
            for k, v in sd.items()}


def _state_dict_bytes(sd: Dict[str, torch.Tensor]) -> int:
    """Rough on-wire byte count of an fp32 state-dict.  Non-tensor
    entries (SpikingJelly memory scalars) contribute 0."""
    total = 0
    for v in sd.values():
        if _is_tensor(v):
            total += v.numel() * v.element_size()
    return total


# ---------------------------------------------------------------------------
# Constellation
# ---------------------------------------------------------------------------

class CloudRemovalConstellation:
    """Top-level v1 orchestrator.

    Parameters
    ----------
    num_planes : int
    sats_per_plane : int
    client_datasets : list[list[Dataset]]
        Shape (num_planes, sats_per_plane).  Each entry is the local
        dataset subset for that satellite.
    args : argparse.Namespace
        Required fields: see task.CloudRemovalSNNTask + intra_plane_iters,
        num_epoch, warmup_epochs, bn_local, lr, wd.
    init_state_dict : dict | None
        Shared initial weights; if None, the first satellite's fresh
        init is broadcast to everyone.
    device : str
    logger : callable | None
    """

    def __init__(self,
                 num_planes: int,
                 sats_per_plane: int,
                 client_datasets: Sequence[Sequence[Dataset]],
                 args,
                 init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
                 device: str = "cuda",
                 logger: Optional[Callable[[str], None]] = None,
                 ):
        assert len(client_datasets) == num_planes
        for row in client_datasets:
            assert len(row) == sats_per_plane

        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.args = args
        self.device = torch.device(device)
        self.log = logger if logger is not None else (lambda m: print(m, flush=True))

        # ---- Build per-satellite tasks (seeded from the first task) ----
        self.log(f"[init] building {num_planes}×{sats_per_plane} "
                 f"= {num_planes * sats_per_plane} satellite tasks")
        self.planes: List[List[CloudRemovalSNNTask]] = []
        seed_sd = init_state_dict
        for p in range(num_planes):
            row: List[CloudRemovalSNNTask] = []
            for s in range(sats_per_plane):
                task = CloudRemovalSNNTask(
                    args=args,
                    local_dataset=client_datasets[p][s],
                    init_state_dict=seed_sd,
                    device=str(self.device),
                )
                if seed_sd is None:
                    seed_sd = task.get_weights(cpu=True)
                row.append(task)
            self.planes.append(row)
        self.init_state_dict = seed_sd
        self.log("[init] satellite tasks ready")

        # ---- RelaySum state --------------------------------------------
        # Relay buffers must live on self.device so they can be `.add_`ed
        # with per-satellite weights (from get_weights(cpu=False)) without
        # a device-mismatch error.
        self._zero = zeros_like_state(seed_sd, device=self.device)
        self.received_relay_weights:    List[List[Dict[str, torch.Tensor]]] = []
        self.transmitted_relay_weights: List[List[Dict[str, torch.Tensor]]] = []
        self.received_relay_counts:    List[List[int]] = []
        self.transmitted_relay_counts: List[List[int]] = []
        self._reset_relay_buffers()

        # ---- Connectivity ----------------------------------------------
        self.connectivity_matrix: np.ndarray = self._chain_topology(num_planes)

        # ---- Metrics bookkeeping ---------------------------------------
        self.round_bytes: List[int] = []
        self.round_train_loss: List[float] = []

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chain_topology(n: int) -> np.ndarray:
        m = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            m[i, i] = 1
            if i > 0:
                m[i, i - 1] = 1
            if i < n - 1:
                m[i, i + 1] = 1
        return m

    def set_connectivity_matrix(self, mat: np.ndarray) -> None:
        assert mat.shape == (self.num_planes, self.num_planes)
        self.connectivity_matrix = mat.astype(np.int32).copy()

    def _reset_relay_buffers(self) -> None:
        self.received_relay_weights = [
            [deepcopy(self._zero) for _ in range(self.num_planes)]
            for _ in range(self.num_planes)
        ]
        self.transmitted_relay_weights = [
            [deepcopy(self._zero) for _ in range(self.num_planes)]
            for _ in range(self.num_planes)
        ]
        self.received_relay_counts    = [[0] * self.num_planes for _ in range(self.num_planes)]
        self.transmitted_relay_counts = [[0] * self.num_planes for _ in range(self.num_planes)]

    # ------------------------------------------------------------------
    # One global round
    # ------------------------------------------------------------------

    def train_one_round(self, aggregation_scheme: str) -> float:
        """Execute one global round of FL.  Returns mean train loss."""
        assert aggregation_scheme in (GOSSIP, RELAYSUM, ALLREDUCE), \
            f"unknown scheme: {aggregation_scheme}"

        bn_local = bool(getattr(self.args, "bn_local", False))
        total_rounds = self.args.num_epoch
        warmup = getattr(self.args, "warmup_epochs", 3)

        # --- 1. intra_plane_iters × (local training + intra-plane avg) ---
        # FLSNN revised_constellation.py:204-205 applies lr*=2.093 for
        # RelaySum to compensate for its delayed-aggregation noise
        # (see v2 §25.11 audit for derivation). Gossip and AllReduce
        # keep their baseline LR (scale=1.0). This matches FLSNN exactly.
        lr_scale = 2.093 if aggregation_scheme == RELAYSUM else 1.0

        round_losses: List[float] = []
        for _ in range(self.args.intra_plane_iters):
            for p in range(self.num_planes):
                for s in range(self.sats_per_plane):
                    loss, _, _ = self.planes[p][s].local_training(
                        total_global_rounds=total_rounds,
                        warmup_rounds=warmup,
                        lr_scale=lr_scale,
                    )
                    round_losses.append(loss)
            # Intra-plane aggregation — BN always averaged within a plane.
            # Rationale (v2 update): under v1's IID partition planes were
            # iso-distribution and averaging BN was trivially safe.  Under
            # v2-A's Dirichlet(α=0.1)-over-source partition the per-sat
            # *marginal* shifts within a plane, but a plane-as-relay-node
            # still needs ONE outgoing weight set, and per-plane BN
            # averaging is the standard FedBN assumption.  bn_local only
            # gates the *inter-plane* step (see _relaysum_step's
            # bn_snapshot/restore), so FedBN semantics are preserved
            # regardless of the intra-plane choice here.
            for p in range(self.num_planes):
                plane_avg = average_state_dicts(
                    [t.get_weights(cpu=False) for t in self.planes[p]],
                    bn_local=False,
                )
                for t in self.planes[p]:
                    t.apply_global_weights(plane_avg, bn_local=False)

        # --- 2. Inter-plane aggregation ---------------------------------
        # After intra-plane avg, every sat in plane p shares the same
        # weights.  Take one representative per plane for inter-plane math.
        intra_plane_weights = [
            self.planes[p][0].get_weights(cpu=False)
            for p in range(self.num_planes)
        ]

        if aggregation_scheme == ALLREDUCE:
            allreduced = average_state_dicts(intra_plane_weights, bn_local=bn_local)
            new_plane_states = [allreduced for _ in range(self.num_planes)]
            bytes_this_round = self.num_planes * _state_dict_bytes(allreduced)

        elif aggregation_scheme == GOSSIP:
            new_plane_states = self._gossip_average(intra_plane_weights, bn_local=bn_local)
            bytes_this_round = 0
            for p in range(self.num_planes):
                neighbours = int(self.connectivity_matrix[p].sum() - 1)
                bytes_this_round += neighbours * _state_dict_bytes(intra_plane_weights[p])

        else:   # RELAYSUM
            new_plane_states = self._relaysum_step(intra_plane_weights, bn_local=bn_local)
            bytes_this_round = 0
            for p in range(self.num_planes):
                neighbours = int(self.connectivity_matrix[p].sum() - 1)
                bytes_this_round += neighbours * _state_dict_bytes(intra_plane_weights[p])

        # --- 3. Broadcast to every satellite ----------------------------
        for p in range(self.num_planes):
            for t in self.planes[p]:
                t.apply_global_weights(new_plane_states[p], bn_local=bn_local)
                t.cleanup_between_rounds()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()      # once per round (not per sat)

        mean_loss = float(np.mean(round_losses)) if round_losses else float("nan")
        self.round_train_loss.append(mean_loss)
        self.round_bytes.append(int(bytes_this_round))
        return mean_loss

    # ------------------------------------------------------------------
    # Inter-plane aggregation kernels
    # ------------------------------------------------------------------

    def _gossip_average(self,
                        intra_plane_weights: Sequence[Dict[str, torch.Tensor]],
                        bn_local: bool = False,
                        ) -> List[Dict[str, torch.Tensor]]:
        n = self.num_planes
        new_states: List[Dict[str, torch.Tensor]] = []
        for p in range(n):
            neigh = [q for q in range(n)
                     if self.connectivity_matrix[p, q] == 1 and q != p]
            participants = [intra_plane_weights[p]] + [intra_plane_weights[q] for q in neigh]
            new_states.append(average_state_dicts(participants, bn_local=bn_local))
        return new_states

    def _relaysum_step(self,
                       intra_plane_weights: Sequence[Dict[str, torch.Tensor]],
                       bn_local: bool = False,
                       ) -> List[Dict[str, torch.Tensor]]:
        """Algorithm 2 of the FLSNN paper.  Operates on persistent
        per-plane relay buffers.

        When bn_local=True, BN-tagged values are snapshotted PRE-aggregation
        and restored POST-aggregation, which enforces FedBN semantics even
        through RelaySum's delayed-delivery buffers.
        """
        n = self.num_planes

        bn_snapshots: Optional[List[Dict[str, torch.Tensor]]] = None
        if bn_local:
            bn_snapshots = []
            for p in range(n):
                bn_snapshots.append({
                    k: v.detach().clone()
                    for k, v in intra_plane_weights[p].items()
                    if is_bn_key(k) and _is_tensor(v)
                })

        # ---- Build tailored outgoing messages ----
        for p in range(n):
            for q in range(n):
                if self.connectivity_matrix[p, q] == 1 and p != q:
                    new_relay = _clone_state(intra_plane_weights[p])
                    new_count = 1
                    for l in range(n):
                        if self.connectivity_matrix[l, p] == 1 and l != q and l != p:
                            src = self.received_relay_weights[p][l]
                            for k in new_relay:
                                if _is_tensor(new_relay[k]) and _is_tensor(src.get(k)):
                                    new_relay[k].add_(src[k])
                            new_count += self.received_relay_counts[p][l]
                    self.transmitted_relay_weights[p][q] = new_relay
                    self.transmitted_relay_counts[p][q] = new_count

        # ---- Receive + aggregate into new plane states ----
        # FLSNN Eq 8 (verified against Golden-Slumber/Decentralized-Satellite-FL-dev
        # revised_constellation.py:340-356): aggregated plane state is
        #     x_p = (sum_of_received + self * (N - received_count)) / N
        # i.e. init agg=0, accumulate received relays, fill the (N - count)
        # "missing" planes with self, then divide by fixed N. This gives
        # the "uniform 1/N average with delayed/stale messages" semantics
        # of Eq 8, NOT the "divide by actual received count" that our v1
        # code previously used (which had the unintended effect of making
        # RelaySum behave like Gossip for the first N-1 rounds — see
        # v2 §25.11 of comprehensive_literature.md for the analysis).
        new_states: List[Dict[str, torch.Tensor]] = []
        for p in range(n):
            # Start from zeros — self-weight is added back explicitly below
            # as (N - received_count) copies, matching FLSNN semantics.
            agg = zeros_like_state(intra_plane_weights[p], device=self.device)
            received_count = 0     # DOES NOT include self
            for q in range(n):
                if self.connectivity_matrix[q, p] == 1 and q != p:
                    self.received_relay_weights[p][q] = self.transmitted_relay_weights[q][p]
                    self.received_relay_counts[p][q]  = self.transmitted_relay_counts[q][p]
                    src = self.received_relay_weights[p][q]
                    for k in agg:
                        if _is_tensor(agg[k]) and _is_tensor(src.get(k)):
                            if agg[k].dtype.is_floating_point:
                                agg[k].add_(src[k])
                            else:
                                agg[k] = agg[k] + src[k]
                    received_count += self.received_relay_counts[p][q]
            # Fill (N - received_count) missing slots with self-weight
            self_w = intra_plane_weights[p]
            fill_factor = float(n - received_count)
            if fill_factor > 0:
                for k in agg:
                    if _is_tensor(agg[k]) and _is_tensor(self_w.get(k)):
                        if agg[k].dtype.is_floating_point:
                            agg[k].add_(self_w[k], alpha=fill_factor)
                        else:
                            agg[k] = agg[k] + self_w[k] * int(fill_factor)
            # Divide by fixed N — not by received_count
            state_div(agg, float(n))

            if bn_snapshots is not None:
                for k, v in bn_snapshots[p].items():
                    dst = agg.get(k)
                    if _is_tensor(dst) and _is_tensor(v):
                        dst.copy_(v)

            new_states.append(agg)

        return new_states

    # ------------------------------------------------------------------
    def snapshot_plane_weights(self) -> List[Dict[str, torch.Tensor]]:
        """CPU copy of current per-plane model weights (for external eval)."""
        return [self.planes[p][0].get_weights(cpu=True) for p in range(self.num_planes)]
