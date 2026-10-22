"""OrbitALIF fancy architecture diagram (replaces the flat fig1.tex).

Generates `fig1.pdf` — a 3D-style block-diagram of the OrbitALIF
single-satellite architecture + the 5×10 Walker-Star federated topology.

Architecture reflects the ACTUAL VLIFNet implementation (vlifnet.py):
  - L1 encoder/decoder: SSHB (Spectro-temporal Spike Hyper-Block)
  - L2/L3 encoder/decoder: DFRB (Dual-Frequency Residual Block)
  - L3 has a DIRECT encoder→decoder connection (no separate bottleneck)
  - Only L1 and L2 have AGFM skip connections (L3 has none)
  - Decoder L1 has TWO SSHB blocks (SSHB-1 + SSHB-2 = additional_sunet_level1)

New nomenclature (§III of paper):
  DFRB  = Dual-Frequency Residual Block       (was MFRB / SRB)
  SSHB  = Spectro-temporal Spike Hyper-Block  (was SUNet_Level1_Block)
  SHAM  = Spectral-Hybrid Attention Module    (was FSTAModule)
  5QS   = 5-level Quantized Spike             (was MultiSpike4)
  AGFM  = Adaptive Gated Fusion Module        (was GatedSkipFusion)
  TCAM  = Temporal-Channel Attention Module   (was TimeAttention / CTM)
  FSE   = Frequency Spectral Enhancement MLP  (was FreMLPBlock)
  TDBN  = Threshold-Dependent Batch Norm      (unchanged)
  VLIF  = Variable-state LIF neuron           (mem_update + 5QS)

Usage
-----
    python -m cloud_removal_v2.plot_arch_diagram --out_dir ./figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, PathPatch
from matplotlib.path import Path as MPath


# ---------------------------------------------------------------------------
# Wong-2011 colour-blind-safe palette  (identical hex codes to fig1.tex)
# ---------------------------------------------------------------------------
C_BLUE   = "#0173B2"
C_ORANGE = "#DE8F05"
C_GREEN  = "#029E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC78BC"
C_YELLOW = "#ECE133"
C_GRAY   = "#4D4D4D"
C_LIGHT  = "#F4F4F4"
C_WHITE  = "#FFFFFF"


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def lighten(color: str, amount: float = 0.55) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return (r + (1-r)*amount, g + (1-g)*amount, b + (1-b)*amount)


def darken(color: str, amount: float = 0.30) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return (r*(1-amount), g*(1-amount), b*(1-amount))


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def shadow_rect(ax, x, y, w, h, radius=0.05, dx=0.04, dy=-0.04, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (x+dx, y+dy), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor="#9A9A9A", edgecolor="none", alpha=0.32, zorder=zorder))


def fancy_block(ax, x, y, w, h, title, subtitle=None, color=C_BLUE, *,
                radius=0.06, fontsize=7.0, subfontsize=5.2, zorder=3,
                bold=True, shadow=True):
    if shadow:
        shadow_rect(ax, x, y, w, h, radius=radius, zorder=zorder-1)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=lighten(color, 0.62), edgecolor=color, linewidth=1.0,
        zorder=zorder))
    cy = y + h/2 + (0.07 if subtitle else 0)
    ax.text(x+w/2, cy, title, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color=darken(color, 0.30), zorder=zorder+1)
    if subtitle:
        ax.text(x+w/2, y+h/2-0.10, subtitle, ha="center", va="center",
                fontsize=subfontsize, color=darken(color, 0.20),
                alpha=0.92, zorder=zorder+1)


def tensor_3d(ax, cx, cy, fw, fh, depth, color=C_BLUE, *,
              label=None, dim_label=None, zorder=2):
    dx, dy = depth*0.65, depth*0.40
    x0, y0, x1, y1 = cx-fw/2, cy-fh/2, cx+fw/2, cy+fh/2
    fc_front = lighten(color, 0.45)
    fc_top   = lighten(color, 0.20)
    fc_right = lighten(color, 0.65)
    ec       = darken(color, 0.20)
    ax.add_patch(Polygon([[x1,y0],[x1+dx,y0+dy],[x1+dx,y1+dy],[x1,y1]],
                          closed=True, facecolor=fc_right, edgecolor=ec,
                          linewidth=0.5, zorder=zorder))
    ax.add_patch(Polygon([[x0,y1],[x1,y1],[x1+dx,y1+dy],[x0+dx,y1+dy]],
                          closed=True, facecolor=fc_top, edgecolor=ec,
                          linewidth=0.5, zorder=zorder))
    ax.add_patch(FancyBboxPatch((x0,y0), fw, fh,
                                 boxstyle="round,pad=0,rounding_size=0.018",
                                 facecolor=fc_front, edgecolor=ec,
                                 linewidth=0.6, zorder=zorder+1))
    if label:
        ax.text(cx, cy+(0.04 if dim_label else 0), label,
                ha="center", va="center", fontsize=4.6, fontweight="bold",
                color=darken(color, 0.40), zorder=zorder+2)
    if dim_label:
        ax.text(cx, cy-0.10, dim_label, ha="center", va="center",
                fontsize=4.0, color=darken(color, 0.30),
                alpha=0.92, zorder=zorder+2)


def arrow_h(ax, x1, x2, y, *, label=None, color=C_GRAY, lw=0.9, zorder=5):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=7), zorder=zorder)
    if label:
        ax.text((x1+x2)/2, y+0.07, label, ha="center", va="bottom",
                fontsize=4.5, color=color, zorder=zorder+1)


def skip_with_gate(ax, x, y_top, y_bot, gate_label="AGFM",
                   color=C_PURPLE, lw=0.9, zorder=4):
    ymid = (y_top + y_bot) / 2
    ax.plot([x,x],[y_top, ymid+0.13], color=color, lw=lw,
            ls=(0,(3,1.5)), zorder=zorder, alpha=0.85)
    ax.plot([x,x],[ymid-0.13, y_bot], color=color, lw=lw,
            ls=(0,(3,1.5)), zorder=zorder, alpha=0.85)
    ax.add_patch(Polygon(
        [[x,ymid+0.11],[x+0.13,ymid],[x,ymid-0.11],[x-0.13,ymid]],
        closed=True, facecolor=lighten(color,0.45), edgecolor=color,
        linewidth=0.8, zorder=zorder+1))
    ax.text(x, ymid, gate_label, ha="center", va="center",
            fontsize=4.4, fontweight="bold",
            color=darken(color,0.35), zorder=zorder+2)
    ax.annotate("", xy=(x, y_bot+0.005), xytext=(x, y_bot+0.10),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=6), zorder=zorder+1)


# ---------------------------------------------------------------------------
# Architecture-pipeline coordinates  (TRUE code structure)
# ---------------------------------------------------------------------------
# Encoder (L→R) and decoder (R→L) SHARE the same X at each level so that
# skip connections are strictly vertical.  There is NO separate bottleneck:
# encoder_level3 feeds DIRECTLY into decoder_level3.
# The decoder has an EXTRA SSHB at X_D1B (additional_sunet_level1).

X_IN  = 0.42    # input / output image icons  (same x, different y rows)
X_PE  = 1.15    # PatchEmbed
X_E1  = 2.10    # L1 enc: SSHB(24)   — shared with Dec-L1① skip target
X_E2  = 3.20    # L2 enc: 2×DFRB(48) — shared with Dec-L2 skip target
X_E3  = 4.40    # L3 enc: 4×DFRB(96) = L3 dec: 2×DFRB(96)  (direct connect)
X_D1B = 1.38    # L1 dec #2: additional_sunet_level1 SSHB
X_HD  = 0.75    # output head (temporal-mean → Conv2d 24→3)

Y_ENC = 4.55
Y_DEC = 2.75

DIM_IN = (0.55, 0.80)
DIM_PE = (0.55, 0.62)
DIM_E1 = (0.80, 1.08)   # SSHB blocks
DIM_E2 = (0.74, 0.90)
DIM_E3 = (0.62, 0.76)
DIM_HD = (0.55, 0.62)

# Feature-map slab sizes (fw, fh, depth) — depth ∝ channel count
FM_RAW = (0.16, 0.55, 0.04)
FM_24  = (0.16, 0.42, 0.10)
FM_48  = (0.13, 0.34, 0.13)
FM_96  = (0.10, 0.26, 0.16)


def _draw_image_icon(ax, x_c, y_c, w, h, color=C_GRAY, label="Cloudy"):
    shadow_rect(ax, x_c-w/2, y_c-h/2, w, h, radius=0.04, zorder=2)
    ax.add_patch(FancyBboxPatch(
        (x_c-w/2, y_c-h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor=lighten(color, 0.85), edgecolor=color, linewidth=0.9,
        zorder=3))
    for off in (-0.18, 0.0, 0.18):
        ax.plot([x_c-w/2+0.06, x_c+w/2-0.06],
                [y_c+off+0.02,  y_c+off-0.10],
                color=color, lw=0.4, alpha=0.35, zorder=4)
    ax.text(x_c, y_c-h/2-0.16, label, ha="center", va="top",
            fontsize=5.2, color=darken(color, 0.10), zorder=5)


# ---------------------------------------------------------------------------
# Panel (a): Architecture pipeline
# ---------------------------------------------------------------------------

def _panel_architecture(ax):
    """TRUE VLIFNet topology: SSHB at L1, DFRB at L2/L3, no bottleneck."""

    ax.text(0.22, 5.30,
            r"(a) OrbitALIF — single-satellite pipeline  "
            r"($T{=}4$ steps, 2.30 M params)",
            ha="left", va="center",
            fontsize=8.0, fontweight="bold", color="black")

    ax.add_patch(FancyBboxPatch(
        (0.12, 2.05), 5.62, 3.15,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0))

    # ── ENCODER ─────────────────────────────────────────────────────────────
    _draw_image_icon(ax, X_IN, Y_ENC, *DIM_IN,
                     color=C_GRAY, label=r"$\mathbf{x}$  (cloudy)")

    tensor_3d(ax, (X_IN+X_PE)/2+0.02, Y_ENC,
              FM_RAW[0], FM_RAW[1], FM_RAW[2], C_GRAY,
              dim_label=r"$3{\times}H{\times}W$")

    fancy_block(ax, X_PE-DIM_PE[0]/2, Y_ENC-DIM_PE[1]/2,
                DIM_PE[0], DIM_PE[1],
                title="Patch", subtitle="Embed", color=C_GRAY,
                fontsize=6.5, subfontsize=5.2)

    tensor_3d(ax, (X_PE+X_E1)/2, Y_ENC,
              FM_24[0], FM_24[1], FM_24[2], C_BLUE,
              dim_label=r"$24{\times}H{\times}W$")

    fancy_block(ax, X_E1-DIM_E1[0]/2, Y_ENC-DIM_E1[1]/2,
                DIM_E1[0], DIM_E1[1],
                title="Enc-L1", subtitle="SSHB\n$C{=}24$",
                color=C_BLUE, fontsize=6.6, subfontsize=5.2)

    tensor_3d(ax, (X_E1+X_E2)/2, Y_ENC,
              FM_48[0], FM_48[1], FM_48[2], C_BLUE,
              dim_label=r"$48{\times}\frac{H}{2}{\times}\frac{W}{2}$")
    ax.text((X_E1+X_E2)/2, Y_ENC+0.52, r"$\downarrow\!2$",
            ha="center", va="center", fontsize=5.2, color=darken(C_BLUE, 0.20))

    fancy_block(ax, X_E2-DIM_E2[0]/2, Y_ENC-DIM_E2[1]/2,
                DIM_E2[0], DIM_E2[1],
                title="Enc-L2", subtitle="2×DFRB\n$C{=}48$",
                color=C_BLUE, fontsize=6.4, subfontsize=4.9)

    tensor_3d(ax, (X_E2+X_E3)/2, Y_ENC,
              FM_96[0], FM_96[1], FM_96[2], C_BLUE,
              dim_label=r"$96{\times}\frac{H}{4}{\times}\frac{W}{4}$")
    ax.text((X_E2+X_E3)/2, Y_ENC+0.48, r"$\downarrow\!2$",
            ha="center", va="center", fontsize=5.2, color=darken(C_BLUE, 0.20))

    fancy_block(ax, X_E3-DIM_E3[0]/2, Y_ENC-DIM_E3[1]/2,
                DIM_E3[0], DIM_E3[1],
                title="Enc-L3", subtitle="4×DFRB\n$C{=}96$",
                color=C_BLUE, fontsize=6.3, subfontsize=4.9)

    # ── "No bottleneck" U-bend (right side) ──────────────────────────────────
    xe3r  = X_E3 + DIM_E3[0]/2
    xbend = xe3r + 0.44
    verts = [(xe3r+0.01, Y_ENC), (xbend, Y_ENC),
             (xbend,     Y_DEC), (xe3r+0.01, Y_DEC)]
    ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO]+[MPath.LINETO]*3),
                           fc="none", ec=C_RED, lw=1.1, zorder=4))
    ax.annotate("", xy=(xe3r+0.01, Y_DEC), xytext=(xbend, Y_DEC),
                arrowprops=dict(arrowstyle="-|>", color=C_RED,
                                lw=0.9, mutation_scale=7), zorder=5)
    ax.text(xbend+0.07, (Y_ENC+Y_DEC)/2,
            "direct\n(no bottleneck)",
            ha="left", va="center",
            fontsize=4.5, color=C_RED, fontstyle="italic", zorder=5)

    # ── DECODER ──────────────────────────────────────────────────────────────
    fancy_block(ax, X_E3-DIM_E3[0]/2, Y_DEC-DIM_E3[1]/2,
                DIM_E3[0], DIM_E3[1],
                title="Dec-L3", subtitle="2×DFRB\n$C{=}96$",
                color=C_ORANGE, fontsize=6.3, subfontsize=4.9)

    tensor_3d(ax, (X_E3+X_E2)/2, Y_DEC,
              FM_48[0], FM_48[1], FM_48[2], C_ORANGE)
    ax.text((X_E3+X_E2)/2, Y_DEC-0.46, r"$\uparrow\!2$",
            ha="center", va="center", fontsize=5.2, color=darken(C_ORANGE, 0.25))

    fancy_block(ax, X_E2-DIM_E2[0]/2, Y_DEC-DIM_E2[1]/2,
                DIM_E2[0], DIM_E2[1],
                title="Dec-L2", subtitle="2×DFRB\n$C{=}48$",
                color=C_ORANGE, fontsize=6.4, subfontsize=4.9)

    tensor_3d(ax, (X_E2+X_E1)/2, Y_DEC,
              FM_24[0], FM_24[1], FM_24[2], C_ORANGE)
    ax.text((X_E2+X_E1)/2, Y_DEC-0.46, r"$\uparrow\!2$",
            ha="center", va="center", fontsize=5.2, color=darken(C_ORANGE, 0.25))

    fancy_block(ax, X_E1-DIM_E1[0]/2, Y_DEC-DIM_E1[1]/2,
                DIM_E1[0], DIM_E1[1],
                title="Dec-L1①", subtitle="SSHB\n$C{=}24$",
                color=C_ORANGE, fontsize=6.4, subfontsize=5.0)

    tensor_3d(ax, (X_E1+X_D1B)/2, Y_DEC,
              FM_24[0], FM_24[1], FM_24[2], C_ORANGE)

    fancy_block(ax, X_D1B-DIM_E1[0]/2, Y_DEC-DIM_E1[1]/2,
                DIM_E1[0], DIM_E1[1],
                title="Dec-L1②", subtitle="SSHB\n$C{=}24$",
                color=C_ORANGE, fontsize=6.4, subfontsize=5.0)
    ax.text(X_D1B, Y_DEC+DIM_E1[1]/2+0.11,
            "extra refinement",
            ha="center", va="bottom", fontsize=4.2,
            color=darken(C_ORANGE, 0.30), fontstyle="italic")

    tensor_3d(ax, (X_D1B+X_HD)/2, Y_DEC,
              FM_24[0], FM_24[1], FM_24[2], C_GRAY)

    fancy_block(ax, X_HD-DIM_HD[0]/2, Y_DEC-DIM_HD[1]/2,
                DIM_HD[0], DIM_HD[1],
                title="Output", subtitle="Head", color=C_GRAY,
                fontsize=6.5, subfontsize=5.2)
    ax.text(X_HD, Y_DEC+DIM_HD[1]/2+0.05,
            r"$\langle\,\cdot\,\rangle_T$",
            ha="center", va="bottom", fontsize=5.5, color=C_GRAY)

    tensor_3d(ax, (X_HD+X_IN)/2-0.02, Y_DEC,
              FM_RAW[0], FM_RAW[1], FM_RAW[2], C_GRAY)

    _draw_image_icon(ax, X_IN, Y_DEC, *DIM_IN,
                     color=C_GRAY, label=r"$\hat{\mathbf{y}}$  (restored)")

    # ── Arrows ───────────────────────────────────────────────────────────────
    for (xa, wa), (xb, wb) in [
        ((X_IN,  DIM_IN[0]),  (X_PE,  DIM_PE[0])),
        ((X_PE,  DIM_PE[0]),  (X_E1,  DIM_E1[0])),
        ((X_E1,  DIM_E1[0]),  (X_E2,  DIM_E2[0])),
        ((X_E2,  DIM_E2[0]),  (X_E3,  DIM_E3[0])),
    ]:
        arrow_h(ax, xa+wa/2+0.01, xb-wb/2-0.01, Y_ENC,
                color=C_GRAY, lw=0.7, zorder=2)

    for (xa, wa), (xb, wb) in [
        ((X_E3,  DIM_E3[0]), (X_E2,  DIM_E2[0])),
        ((X_E2,  DIM_E2[0]), (X_E1,  DIM_E1[0])),
        ((X_E1,  DIM_E1[0]), (X_D1B, DIM_E1[0])),
        ((X_D1B, DIM_E1[0]), (X_HD,  DIM_HD[0])),
        ((X_HD,  DIM_HD[0]), (X_IN,  DIM_IN[0])),
    ]:
        ax.annotate("",
                    xy=(xb+wb/2+0.01, Y_DEC),
                    xytext=(xa-wa/2-0.01, Y_DEC),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                    lw=0.7, mutation_scale=7), zorder=2)

    # ── Skip connections — ONLY L1 and L2 (L3 has NO skip) ──────────────────
    skip_with_gate(ax, x=X_E2,
                   y_top=Y_ENC - DIM_E2[1]/2 - 0.02,
                   y_bot=Y_DEC + DIM_E2[1]/2 + 0.02,
                   gate_label="AGFM", color=C_PURPLE, lw=0.8)

    skip_with_gate(ax, x=X_E1,
                   y_top=Y_ENC - DIM_E1[1]/2 - 0.02,
                   y_bot=Y_DEC + DIM_E1[1]/2 + 0.02,
                   gate_label="AGFM", color=C_PURPLE, lw=0.8)

    # Global residual
    ax.annotate("",
                xy=(X_IN-0.04, Y_DEC+DIM_IN[1]/2-0.05),
                xytext=(X_IN-0.04, Y_ENC-DIM_IN[1]/2+0.05),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                lw=0.6, ls=(0,(3,1.5)), mutation_scale=6),
                zorder=2)
    ax.text(X_IN-0.18, (Y_ENC+Y_DEC)/2,
            "global\nresidual", ha="right", va="center",
            fontsize=4.2, color=C_GRAY, zorder=2)


# ---------------------------------------------------------------------------
# Panel (b): Federated topology
# ---------------------------------------------------------------------------

def _panel_constellation(ax):
    ax.text(7.50, 5.30, r"(b) 5×10 Walker-Star federation",
            ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="black")

    ax.add_patch(FancyBboxPatch(
        (6.40, 2.05), 2.25, 3.15,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0))

    n_sat, n_planes = 10, 5
    ring_rx, ring_ry = 0.72, 0.20
    cx_const = 7.50
    plane_ys = [4.82 + (2.40-4.82)*i/(n_planes-1) for i in range(n_planes)]
    plane_colors = [C_BLUE, C_GREEN, C_PURPLE, C_ORANGE, C_RED]
    sat_positions = []

    for p, (cy, pc) in enumerate(zip(plane_ys, plane_colors)):
        ax.add_patch(mpatches.Ellipse(
            (cx_const, cy), 2*ring_rx, 2*ring_ry,
            facecolor=lighten(pc, 0.85), edgecolor=pc, linewidth=0.7, zorder=2))
        ax.text(cx_const-ring_rx-0.08, cy,
                rf"$\mathcal{{P}}_{{{p}}}$",
                ha="right", va="center", fontsize=5.2,
                color=darken(pc, 0.20), fontweight="bold")
        sats = []
        for i in range(n_sat):
            ang = 2*math.pi*i/n_sat
            sx, sy = cx_const+ring_rx*math.cos(ang), cy+ring_ry*math.sin(ang)
            sats.append((sx, sy))
            ax.add_patch(mpatches.Circle((sx, sy),
                0.050 if i==0 else 0.035,
                facecolor=lighten(pc,0.55) if i==0 else "white",
                edgecolor=pc if i==0 else darken(pc,0.10),
                linewidth=0.8 if i==0 else 0.5, zorder=4))
        sat_positions.append(sats)
        for i in range(n_sat):
            j = (i+1)%n_sat
            ax.plot([sats[i][0], sats[j][0]], [sats[i][1], sats[j][1]],
                    color=lighten(pc, 0.30), lw=0.4, alpha=0.55, zorder=3)

    for p in range(n_planes-1):
        x1p, y1p = sat_positions[p][0]
        x2p, y2p = sat_positions[p+1][0]
        midx, midy = (x1p+x2p)/2+0.18, (y1p+y2p)/2
        ax.plot([x1p, midx, x2p], [y1p, midy, y2p],
                color=C_RED, lw=0.7, alpha=0.85, zorder=5)
        ax.annotate("", xy=(x2p+0.005, y2p), xytext=(midx, midy),
                    arrowprops=dict(arrowstyle="-|>", color=C_RED,
                                    lw=0.5, mutation_scale=4), zorder=5)

    x0, y0p = sat_positions[0][0]
    x4, y4  = sat_positions[-1][0]
    ax.plot([x0+0.35, x4+0.35], [y0p, y4],
            color=lighten(C_RED, 0.30), lw=0.5, ls=(0,(2,1.5)),
            alpha=0.7, zorder=4)

    leg_y = 2.28
    for sx, sz, fc, ec, lbl in [
        (6.55, 0.045, lighten(C_BLUE,0.55), C_BLUE, "head"),
        (7.04, 0.030, "white",              C_GRAY, "sat"),
    ]:
        ax.add_patch(mpatches.Circle((sx, leg_y), sz, facecolor=fc, edgecolor=ec,
                                     linewidth=0.7, zorder=3))
        ax.text(sx+0.07, leg_y, lbl, ha="left", va="center",
                fontsize=4.5, color=C_GRAY)
    ax.plot([7.34,7.48],[leg_y,leg_y], color=lighten(C_BLUE,0.30), lw=0.7, alpha=0.85)
    ax.text(7.52, leg_y, "intra",  ha="left", va="center", fontsize=4.5, color=C_GRAY)
    ax.plot([7.82,7.96],[leg_y,leg_y], color=C_RED, lw=0.7)
    ax.text(8.00, leg_y, "Gossip", ha="left", va="center", fontsize=4.5, color=C_GRAY)
    ax.text(7.50, 2.15, r"FedBN + Dir$(\alpha{=}0.1)$",
            ha="center", va="center", fontsize=4.8,
            fontstyle="italic", color=darken(C_GRAY, 0.10))


# ---------------------------------------------------------------------------
# Panel (c): DFRB module zoom-in
# ---------------------------------------------------------------------------

def _panel_module_dfrb(ax, x0, y0, w, h):
    ax.add_patch(FancyBboxPatch((x0,y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0))
    ax.text(x0+0.10, y0+h-0.14, "(c) DFRB",
            ha="left", va="center", fontsize=7.0, fontweight="bold", color="black")
    ax.text(x0+w-0.08, y0+h-0.14, "(Dual-Frequency Residual Block)",
            ha="right", va="center", fontsize=4.8, color=C_GRAY, fontstyle="italic")

    cx_in   = x0+0.28
    cx_lif  = x0+0.80
    cx_conv = x0+1.32
    cx_bn   = x0+1.78
    cx_gate = x0+2.18
    cx_attn = x0+2.56
    cx_out  = x0+2.90
    y_top = y0+h-0.52
    y_bot = y0+0.42
    y_mid = (y_top+y_bot)/2

    fancy_block(ax, cx_in-0.18, y_mid-0.16, 0.36, 0.32,
                title=r"$\mathbf{X}$", color=C_GRAY, fontsize=6.0, shadow=False)

    for (y, lif_lbl) in [(y_top, "5QS-LIF₁"), (y_bot, "PS-LIF₂")]:
        lw = 0.44 if y == y_bot else 0.40
        fancy_block(ax, cx_lif-lw, y-0.13, lw*2, 0.26,
                    title=lif_lbl, color=C_YELLOW, fontsize=5.2, radius=0.04)
        fancy_block(ax, cx_conv-0.22, y-0.13, 0.44, 0.26,
                    title="Conv", color=C_BLUE, fontsize=5.5, radius=0.04)
        fancy_block(ax, cx_bn-0.18, y-0.13, 0.36, 0.26,
                    title="TDBN", color=C_GREEN, fontsize=5.0, radius=0.04)

    ax.add_patch(mpatches.Circle((cx_gate, y_mid), 0.14,
                facecolor=lighten(C_PURPLE,0.45), edgecolor=C_PURPLE,
                linewidth=0.8, zorder=4))
    ax.text(cx_gate, y_mid, r"$\sigma(g)$",
            ha="center", va="center", fontsize=4.8,
            color=darken(C_PURPLE,0.30), zorder=5)

    fancy_block(ax, cx_attn-0.18, y_mid-0.24, 0.36, 0.48,
                title="TCS-\nAtt", color=C_ORANGE, fontsize=5.0,
                radius=0.04, shadow=False)
    ax.text(cx_attn, y_mid-0.35, "SHAM",
            ha="center", va="top", fontsize=4.0, color=darken(C_GREEN, 0.20))

    fancy_block(ax, cx_out-0.18, y_mid-0.16, 0.36, 0.32,
                title=r"$\mathbf{Y}$", color=C_GRAY, fontsize=6.0, shadow=False)

    # Input split
    for sign, y_path in [(+1, y_top), (-1, y_bot)]:
        y_offset = 0.05 * sign
        ax.plot([cx_in+0.18, cx_lif-0.40], [y_mid+y_offset]*2, color=C_GRAY, lw=0.5)
        ax.plot([cx_lif-0.40]*2, [y_mid+y_offset, y_path-0.05*sign], color=C_GRAY, lw=0.5)
        ax.annotate("", xy=(cx_lif-0.40+(0.20 if y_path==y_top else 0.22), y_path),
                    xytext=(cx_lif-0.40, y_path),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5, mutation_scale=4))

    for y in (y_top, y_bot):
        ax.annotate("", xy=(cx_conv-0.22, y), xytext=(cx_lif+0.20, y),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5, mutation_scale=4))
        ax.annotate("", xy=(cx_bn-0.18, y), xytext=(cx_conv+0.22, y),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5, mutation_scale=4))

    ax.plot([cx_bn+0.18, cx_gate-0.04], [y_top, y_mid+0.10], color=C_PURPLE, lw=0.5)
    ax.plot([cx_bn+0.18, cx_gate-0.04], [y_bot, y_mid-0.10], color=C_PURPLE, lw=0.5)
    ax.annotate("", xy=(cx_attn-0.18, y_mid), xytext=(cx_gate+0.14, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=0.6, mutation_scale=5))
    ax.annotate("", xy=(cx_out-0.18, y_mid), xytext=(cx_attn+0.18, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=0.6, mutation_scale=5))

    ax.annotate("", xy=(cx_out-0.20, y0+0.20), xytext=(cx_in+0.16, y0+0.20),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.4,
                                ls=(0,(2.5,1.5)), mutation_scale=4))
    ax.text((cx_in+cx_out)/2, y0+0.10, "shortcut  (Conv+TDBN)",
            ha="center", va="center", fontsize=4.2, color=C_GRAY, fontstyle="italic")

    ax.text(cx_lif-0.30, y_top+0.18, "temporal HP",
            ha="center", va="center", fontsize=4.3,
            color=darken(C_GRAY,0.10), fontstyle="italic")
    ax.text(cx_lif-0.30, y_bot-0.18, "spatial HP",
            ha="center", va="center", fontsize=4.3,
            color=darken(C_GRAY,0.10), fontstyle="italic")


# ---------------------------------------------------------------------------
# Panel (d): 5QS step function
# ---------------------------------------------------------------------------

def _panel_module_5qs(ax, x0, y0, w, h):
    ax.add_patch(FancyBboxPatch((x0,y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0))
    ax.text(x0+0.10, y0+h-0.14, "(b) 5QS",
            ha="left", va="center", fontsize=7.0, fontweight="bold", color="black")
    ax.text(x0+w-0.08, y0+h-0.14, "(5-level Quantized Spike)",
            ha="right", va="center", fontsize=4.8, color=C_GRAY, fontstyle="italic")

    x_ax0, x_ax1 = x0+0.28, x0+w-0.18
    y_ax0, y_ax1 = y0+0.30, y0+h-0.40

    for (xy, xy2, text) in [
        ((x_ax1, y_ax0), (x_ax0, y_ax0), None),
        ((x_ax0, y_ax1), (x_ax0, y_ax0), None),
    ]:
        ax.annotate("", xy=xy, xytext=xy2,
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=0.7, mutation_scale=5))
    ax.text(x_ax1+0.04, y_ax0, r"$u$", ha="left", va="center",
            fontsize=5.0, color=C_GRAY)
    ax.text(x_ax0, y_ax1+0.06, r"$s$", ha="center", va="bottom",
            fontsize=5.0, color=C_GRAY)

    ux = lambda u: x_ax0 + u/4*(x_ax1-x_ax0)
    sy = lambda s: y_ax0 + s*(y_ax1-y_ax0)

    ax.add_patch(mpatches.Rectangle((ux(0),sy(0)), ux(4)-ux(0), sy(1)-sy(0),
                facecolor=lighten(C_PURPLE,0.70), edgecolor="none", alpha=0.45, zorder=1))

    step_u = [0.5, 1.5, 2.5, 3.5]
    step_s = [0.25, 0.50, 0.75, 1.0]
    pts = [(ux(0),sy(0)),(ux(0.5),sy(0))]
    for k,(uj,sn) in enumerate(zip(step_u, step_s)):
        pts.append((ux(uj), sy(sn)))
        pts.append((ux(step_u[k+1] if k+1<4 else 4), sy(sn)))
    ax.plot(*zip(*pts), color=C_BLUE, lw=1.2, zorder=4)
    for uj,sn in zip(step_u, step_s):
        ax.add_patch(mpatches.Circle((ux(uj),sy(sn)), 0.012,
                    facecolor="white", edgecolor=C_BLUE, linewidth=0.6, zorder=5))

    ax.plot([ux(0),ux(0.5),ux(0.5),ux(4)],[sy(0),sy(0),sy(1),sy(1)],
            color=C_ORANGE, lw=0.8, ls=(0,(2.5,1.5)), zorder=3, alpha=0.9)

    ax.text(x_ax1-0.05, sy(0.30), "5QS", ha="right", va="center",
            fontsize=4.7, color=darken(C_BLUE,0.20))
    ax.text(x_ax1-0.05, sy(0.18), "binary", ha="right", va="center",
            fontsize=4.7, color=darken(C_ORANGE,0.20))
    ax.text((x_ax0+x_ax1)/2, y_ax0-0.16,
            r"surrogate $u{\in}[0,4]$, $\partial s/\partial u{=}1/4$",
            ha="center", va="center", fontsize=4.3,
            color=darken(C_PURPLE,0.30), fontstyle="italic")

    for u,lbl in [(0,"0"),(4,"4")]:
        ax.text(ux(u), y_ax0-0.05, lbl, ha="center", va="top",
                fontsize=4.4, color=C_GRAY)
    for s,lbl in [(0,"0"),(1,"1")]:
        ax.text(x_ax0-0.03, sy(s), lbl, ha="right", va="center",
                fontsize=4.4, color=C_GRAY)


# ---------------------------------------------------------------------------
# Panel (e): SHAM attention module
# ---------------------------------------------------------------------------

def _panel_module_sham(ax, x0, y0, w, h):
    ax.add_patch(FancyBboxPatch((x0,y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0))
    ax.text(x0+0.10, y0+h-0.14, "(c) SHAM",
            ha="left", va="center", fontsize=7.0, fontweight="bold", color="black")
    ax.text(x0+w-0.08, y0+h-0.14, "(Spectral-Hybrid Attention)",
            ha="right", va="center", fontsize=4.8, color=C_GRAY, fontstyle="italic")

    y_t_in  = y0+h-0.30
    y_t_w   = y0+h-0.55
    y_t_out = y0+h-0.78
    y_s_in  = y0+0.78
    y_s_w   = y0+0.50
    y_s_out = y0+0.22
    cx0     = x0+0.42
    dx, nc  = 0.16, 4

    ax.text(cx0+1.5*dx, y_t_in+0.10, "TAA (temporal amp.)",
            ha="center", va="bottom",
            fontsize=4.6, fontstyle="italic", color=darken(C_GRAY,0.10))
    for t in range(nc):
        tensor_3d(ax, cx0+t*dx, y_t_in, 0.10, 0.10, 0.05, C_BLUE, zorder=2)
    ax.text(cx0+nc*dx+0.05, y_t_in, rf"$T={nc}$",
            ha="left", va="center", fontsize=4.5, color=darken(C_GRAY,0.10))

    wcs = [C_ORANGE, C_BLUE, C_GREEN, C_PURPLE]
    for t, wc in enumerate(wcs):
        ax.add_patch(mpatches.Rectangle((cx0+t*dx-0.05, y_t_w-0.05), 0.10, 0.10,
                    facecolor=lighten(wc,0.30), edgecolor=darken(wc,0.20),
                    linewidth=0.4, zorder=3))
    ax.annotate("", xy=(cx0+1.5*dx, y_t_w+0.07), xytext=(cx0+1.5*dx, y_t_in-0.07),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5, mutation_scale=4))
    ax.text(cx0+1.5*dx+0.04, (y_t_w+y_t_in)/2, "FC(T)",
            fontsize=4.0, color=C_GRAY, ha="left", va="center")
    for t, wc in enumerate(wcs):
        tensor_3d(ax, cx0+t*dx, y_t_out, 0.10, 0.10, 0.05, wc, zorder=2)
    ax.annotate("", xy=(cx0+1.5*dx, y_t_out+0.07), xytext=(cx0+1.5*dx, y_t_w-0.07),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5, mutation_scale=4))
    ax.text(cx0+1.5*dx+0.04, (y_t_w+y_t_out)/2,
            r"$\sigma(\cdot)$,Bcast$\times$",
            fontsize=4.0, color=C_GRAY, ha="left", va="center")
    # TAA residual: x' = x · t_gate + x  (fsta_module.py line 123)
    ax.text(cx0+nc*dx+0.05, y_t_out, r"$+\,\mathbf{x}$",
            ha="left", va="center", fontsize=4.6,
            color=darken(C_PURPLE,0.30), fontweight="bold")

    ax.plot([x0+0.10, x0+w-0.10], [(y_t_out+y_s_in)/2]*2,
            color="#D0D0D0", lw=0.4, ls=(0,(1,1.2)))

    ax.text(x0+0.10, y_s_in+0.12, "SFA (spectral freq. attn.)",
            fontsize=4.6, fontstyle="italic", color=darken(C_GRAY,0.10))
    tensor_3d(ax, cx0+0.55, y_s_in, 0.30, 0.18, 0.10, C_BLUE, zorder=2)
    ax.text(cx0+0.80, y_s_in, r"$\mathbf{U}$",
            ha="left", va="center", fontsize=5.0, color=darken(C_BLUE,0.30))

    gcx, gcy = cx0+0.55, y_s_w
    for i in range(3):
        for j in range(3):
            ax.add_patch(mpatches.Rectangle(
                (gcx-0.10+j*0.07, gcy-0.10+i*0.07), 0.06, 0.06,
                facecolor=lighten(C_BLUE, 0.50+0.20*((i+j)%3)/2),
                edgecolor=darken(C_BLUE,0.20), linewidth=0.3, zorder=3))

    ax.annotate("", xy=(gcx, gcy+0.10), xytext=(gcx, y_s_in-0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5, mutation_scale=4))
    ax.text(gcx+0.04, (gcy+y_s_in)/2, "rfft2",
            fontsize=4.0, color=C_GRAY, ha="left", va="center")

    # Reweighted spectrum (FreqMLP output: mag * mag_weight)
    ax.add_patch(mpatches.Rectangle((gcx-0.18, y_s_out-0.07), 0.36, 0.14,
                facecolor=lighten(C_BLUE,0.30), edgecolor=darken(C_BLUE,0.30),
                linewidth=0.5, hatch="//", zorder=3, alpha=0.85))
    ax.annotate("", xy=(gcx, y_s_out+0.08), xytext=(gcx, gcy-0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5, mutation_scale=4))
    ax.text(gcx+0.04, (gcy+y_s_out)/2, r"FreqMLP$\cdot$",
            fontsize=4.0, color=C_GRAY, ha="left", va="center")

    # Post-frequency stages (DCTSpatialAttention.forward steps 5-7):
    #   iFFT (preserve phase) → 7x7 conv → spatial mask [B,1,H,W]
    #   → broadcast over T-steps → multiply with x → +x residual
    ax.text(gcx+0.20, y_s_out, r"$\to$ iFFT $\to$ 7$\times$7 Conv",
            ha="left", va="center", fontsize=3.6,
            color=darken(C_GRAY, 0.10), fontstyle="italic")
    ax.text(gcx+0.20, y_s_out-0.13,
            r"$\to$ mask $\cdot$ $\mathbf{x}_{[T,B,C,H,W]}$ $+\,\mathbf{x}$",
            ha="left", va="center", fontsize=3.6,
            color=darken(C_PURPLE, 0.30), fontweight="bold")

    ax.text(x0+w-0.08, y0+0.04,
            r"out$=\mathbf{x}+\sigma(s)(\mathbf{y}-\mathbf{x}),\ s{=}0$",
            ha="right", va="bottom", fontsize=4.0,
            color=darken(C_PURPLE,0.30), fontstyle="italic")


# ---------------------------------------------------------------------------
# Panel (f): SSHB module detail  (vertical flowchart)
# ---------------------------------------------------------------------------

def _panel_module_sshb(ax, x0, y0, w, h):
    ax.add_patch(FancyBboxPatch((x0,y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0))
    ax.text(x0+0.10, y0+h-0.14, "(d) SSHB",
            ha="left", va="center", fontsize=7.0, fontweight="bold", color="black")
    ax.text(x0+w-0.08, y0+h-0.14,
            "(Spectro-temporal Spike Hyper-Block)",
            ha="right", va="center", fontsize=4.6, color=C_GRAY, fontstyle="italic")

    cx   = x0 + w*0.36
    bw   = w*0.56
    bh   = 0.18

    steps = [
        (r"DFRB  ($C$)",                  C_BLUE),
        (r"PixelUnshuffle \xd72",         C_GRAY),
        (r"5QS-LIF  $[T{\times}4{=}16]$",  C_YELLOW),
        (r"TCAM",                          C_GREEN),
        (r"Conv3d  $16{\to}T{=}4$",       C_GRAY),
        (r"5QS-LIF{\to}Conv{\to}TDBN{\to}TCS-Att  \xd72",C_YELLOW),
        (r"bilinear $\uparrow\!2$",        C_GRAY),
        (r"TCS-Att",                       C_ORANGE),
        (r"FSE-MLP",                       C_GREEN),
    ]
    n = len(steps)
    y_top_blk = y0+h-0.42
    gap_v     = (y_top_blk - (y0+0.24)) / (n-1)
    ys = [y_top_blk - i*gap_v for i in range(n)]

    for i, (lbl, col) in enumerate(steps):
        fancy_block(ax, cx-bw/2, ys[i]-bh/2, bw, bh,
                    title=lbl, color=col,
                    fontsize=4.7, radius=0.03, shadow=(i in (0,4,8)))
        if i < n-1:
            ax.annotate("", xy=(cx, ys[i+1]+bh/2+0.01),
                        xytext=(cx, ys[i]-bh/2-0.01),
                        arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                        lw=0.5, mutation_scale=4), zorder=4)

    ax.text(cx, ys[0]+bh/2+0.09,
            r"$\mathbf{X}\ [T,B,C,H,W]$",
            ha="center", va="bottom", fontsize=4.6,
            color=darken(C_BLUE,0.20), fontweight="bold")
    ax.text(cx, ys[-1]-bh/2-0.09,
            r"$\mathbf{Y}\ [T,B,C,H,W]$",
            ha="center", va="top", fontsize=4.6,
            color=darken(C_ORANGE,0.20), fontweight="bold")

    # Skip branch: DFRB output → ⊕ at bilinear output
    sx   = cx + bw/2 + 0.10
    y_br = ys[0]
    y_rj = ys[6]
    ax.add_patch(PathPatch(
        MPath([(cx+bw/2, y_br),(sx, y_br),(sx, y_rj),(cx+bw/2, y_rj)],
              [MPath.MOVETO]+[MPath.LINETO]*3),
        fc="none", ec=C_PURPLE, lw=0.7, ls=(0,(3,1.5)), zorder=3))
    ax.annotate("", xy=(cx+bw/2, y_rj), xytext=(sx+0.01, y_rj),
                arrowprops=dict(arrowstyle="-|>", color=C_PURPLE,
                                lw=0.6, mutation_scale=4), zorder=4)
    ax.text(cx+bw/2-0.04, y_rj+0.08, "⊕",
            ha="center", va="center", fontsize=7, color=C_PURPLE, zorder=5)
    ax.text(sx+0.06, (y_br+y_rj)/2, "skip\n(DFRB out)",
            ha="left", va="center", fontsize=3.8, color=C_PURPLE, fontstyle="italic")

    ax.text(cx-bw/2-0.05, (ys[1]+ys[2])/2, r"$H/2$",
            ha="right", va="center", fontsize=3.8, color=C_GRAY, fontstyle="italic")
    ax.text(cx-bw/2-0.05, (ys[5]+ys[6])/2, r"${\uparrow}H$",
            ha="right", va="center", fontsize=3.8, color=C_GRAY, fontstyle="italic")


# ---------------------------------------------------------------------------
# Bottom strip: four module zoom-in panels
# ---------------------------------------------------------------------------

def _panel_module_zoom(ax):
    """Bottom strip: 3 zoom-in panels in §III order — 5QS, SHAM, SSHB."""
    bottom_y = 0.05
    panel_h  = 1.90
    margin   = 0.15
    gap      = 0.12
    total_w  = 8.80 - 2*margin - 2*gap    # 3 panels → 2 gaps

    # Relative widths preserve the original 19:24:27 ratio (sum 70).
    w_5qs  = total_w * 0.27
    w_sham = total_w * 0.34
    w_sshb = total_w * 0.39

    x_5qs  = margin
    x_sham = x_5qs  + w_5qs  + gap
    x_sshb = x_sham + w_sham + gap

    _panel_module_5qs (ax, x_5qs,  bottom_y, w_5qs,  panel_h)
    _panel_module_sham(ax, x_sham, bottom_y, w_sham, panel_h)
    _panel_module_sshb(ax, x_sshb, bottom_y, w_sshb, panel_h)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir",  type=str, default="./figures")
    p.add_argument("--out_name", type=str, default="fig1.pdf")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.80, 6.00))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8.80)
    ax.set_ylim(0, 6.00)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    _panel_architecture(ax)
    _panel_module_zoom(ax)

    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
