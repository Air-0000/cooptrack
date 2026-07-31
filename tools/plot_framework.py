"""
Generate Figure 1: Adaptive Fusion framework architecture.
Simple, clean layout with large gaps. Each element is a matplotlib
FancyBboxPatch with embedded text.
"""

import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

OUT_PDF = "fig/framework.pdf"
OUT_PNG = "fig/framework.png"
DPI = 150

W, H = 14, 10  # inches

# Colors
C_VEH  = "#E8E8E8"
C_DRN  = "#E8E8E8"
C_PROT = "#F2F2F2"
C_DGC  = "#FCE4D6"
C_CAA  = "#DDEBF7"
C_ACM  = "#FFF2CC"
C_FUSE = "#E2EFDA"
C_GRU  = "#EDEDED"
C_ARROW = "#404040"

def rbox(ax, x, y, w, h, text, fc="#eee", ec="#666", fs=10, lw=1.0, color="#000"):
    """Rounded rect with centred text."""
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size=0.08",
        linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color=color, fontweight='normal')

def tbox(ax, x, y, w, h, text, fc="#eee", ec="#666", fs=10, lw=1.0, color="#000"):
    """Rounded rect with top-aligned label and centre text."""
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size=0.08",
        linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    # label band at top
    ax.text(x + w/2, y + h - 0.12, text[0], ha='center', va='top',
            fontsize=9, color="#555", fontweight='bold')
    ax.text(x + w/2, y + h/2, text[1], ha='center', va='center',
            fontsize=fs, color=color)

def arrow(ax, p0, p1, color=C_ARROW, lw=1.5, style='-'):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>',
        mutation_scale=20, color=color, linewidth=lw, linestyle=style,
        connectionstyle='arc3,rad=0')
    ax.add_patch(a)

# ----------
fig, ax = plt.subplots(1, 1, figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect('equal'); ax.axis('off')

# ===== Title =====
ax.text(W/2, 9.7, "Adaptive Fusion Framework: DGC + UGIM (+ Stage-3 Dual GRU)",
        ha='center', va='center', fontsize=14, fontweight='bold')

# ===== ROW 1: Inputs (y=8.0–9.2) =====
rbox(ax, 0.4, 8.3, 2.8, 0.8, "Vehicle queries\nFv , Pv",  fc=C_VEH, fs=10)
rbox(ax, 0.4, 7.1, 2.8, 0.8, "Drone queries\nFd , Pd",    fc=C_DRN, fs=10)
rbox(ax, 4.0, 8.3, 1.8, 0.8, "Altitude h",  fc="#eee", fs=10)
rbox(ax, 4.0, 7.1, 1.8, 0.8, "Pose [R|t]",   fc="#eee", fs=10)

# ===== ROW 2: Coordinate Protocol + DGC (y=5.8–6.8) =====
rbox(ax, 6.5, 8.0, 3.0, 0.7, "P_{d-v} = R·Pd + t\n(ego frame)", fc=C_PROT, fs=10)
ax.text(8.0, 7.7, "Coordinate Protocol §III", ha='center', va='top', fontsize=8, color="#666")

rbox(ax, 10.2, 8.3, 3.2, 0.7, "Sinusoidal PE(h), PE(Δh)", fc="#fff", fs=10)
rbox(ax, 10.2, 7.1, 3.2, 0.7, "MLP → ΔP(h) ∈ R³", fc=C_DGC, ec="#C55A11", fs=11, color="#C55A11")
ax.text(11.8, 6.9, "DGC", ha='center', va='top', fontsize=13, fontweight='bold', color="#C55A11")

# ===== Correction tag (between rows) =====
ax.text(7.8, 6.55, "P'd = P_{d-v} + ΔP(h)", ha='center', va='center',
        fontsize=11, color="#222",
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb', lw=0.8))

# ===== ROW 3: UGIM (y=4.2–5.2) =====
rbox(ax, 4.5, 4.6, 3.8, 1.1,
    "CAA:\nS_match = S_sem · S_geo\n(Hungarian on corrected coords)",
    fc=C_CAA, ec="#2E75B6", fs=10, color="#2E75B6")
rbox(ax, 9.5, 4.6, 3.8, 1.1,
    "ACM:\nc = σ( φ_conf([Δp ; P'd]) )\n(displacement-only, IoU-supervised)",
    fc=C_ACM, ec="#BF8F00", fs=10, color="#BF8F00")

ax.text(6.4, 4.4, "UGIM", ha='center', va='top', fontsize=13, fontweight='bold', color="#2E75B6")
ax.text(11.4, 4.4, "ACM (decoupled)", ha='center', va='top', fontsize=12, fontweight='bold', color="#BF8F00")

# ===== Stage-3 Dual GRU (bottom-left of UGIM row, same y) =====
rbox(ax, 0.4, 4.6, 1.6, 0.8, "GRU_veh", fc=C_GRU, ec="#999", fs=10, color="#666")
rbox(ax, 2.3, 4.6, 1.6, 0.8, "GRU_inf", fc=C_GRU, ec="#999", fs=10, color="#666")
ax.text(2.15, 4.35, "Dual GRU (Stage 3 only)\nseparate params per agent",
        ha='center', va='top', fontsize=9, color="#666")

# ===== ROW 4: Fusion (y=2.3–3.4) =====
rbox(ax, 4.5, 2.4, 8.8, 0.9,
    "F_fused = α·(1−c)·F'd + (1−α)·Fv\n(weighted + confidence-gated)",
    fc=C_FUSE, ec="#548235", fs=12, color="#548235")

# ===== ROW 5: Output (y=1.0–1.8) =====
rbox(ax, 4.5, 1.0, 8.8, 0.7, "Fused detection / tracking outputs",
    fc="#fff", ec="#666", fs=11)

# ===== Arrows =====
arrow(ax, (3.2, 8.7), (6.5, 8.35))
arrow(ax, (3.2, 7.5), (6.5, 8.15))
arrow(ax, (5.8, 8.7), (10.2, 8.65))
arrow(ax, (5.8, 7.5), (6.5, 8.15))
arrow(ax, (9.5, 8.35), (10.2, 8.65))
arrow(ax, (11.8, 8.3), (11.8, 7.8))
arrow(ax, (10.2, 7.5), (7.8, 6.8))
arrow(ax, (11.8, 7.1), (11.4, 5.7), color="#BF8F00", style='--')

arrow(ax, (8.3, 5.2), (8.3, 4.2))
arrow(ax, (6.4, 4.6), (6.4, 3.3))
arrow(ax, (11.4, 4.6), (11.4, 3.3))
arrow(ax, (3.7, 4.6), (3.7, 2.85))
arrow(ax, (8.8, 2.4), (8.8, 1.7))

# CAA -> ACM: matched idx
arrow(ax, (8.3, 5.15), (9.5, 5.15), color="#BF8F00", style='--')
ax.text(8.9, 5.30, "matched idx", ha='center', va='bottom', fontsize=8, color="#BF8F00")

# ===== Stage bars =====
def sbar(box_x, box_w, text, yy):
    x0, x1 = box_x, box_x + box_w
    ax.plot([x0, x1], [yy, yy], color="#bbb", lw=1, ls='--')
    ax.text((x0+x1)/2, yy-0.15, text, ha='center', va='top', fontsize=8, color="#888")

sbar(0.2, 13.6, "Inputs | Coordinate protocol | DGC  (always-on core)", 6.9)
sbar(0.2, 13.6, "UGIM: CAA matching + decoupled ACM confidence", 4.2)
sbar(0.2, 4.2, "Stage-3 auxiliary (off by default)", 5.8)
sbar(4.2, 13.6, "Fusion + output", 1.9)

# ===== Legend =====
leg = [
    Rectangle((0,0),1,1,fc=C_DGC,ec="#C55A11",label="DGC (altitude correction)"),
    Rectangle((0,0),1,1,fc=C_CAA,ec="#2E75B6",label="CAA (semantic+geometric matching)"),
    Rectangle((0,0),1,1,fc=C_ACM,ec="#BF8F00",label="ACM (decoupled confidence)"),
    Rectangle((0,0),1,1,fc=C_FUSE,ec="#548235",label="Fusion"),
    Rectangle((0,0),1,1,fc=C_GRU,ec="#999",label="Stage-3 auxiliary"),
]
ax.legend(handles=leg, loc='lower left', bbox_to_anchor=(0, 0.0),
          ncol=5, fontsize=9, frameon=False, handlelength=1.5, columnspacing=1.5)
ax.text(W/2, 0.20, "Solid arrows: data flow.  Dashed: decoupled ACM branch and Stage-3 auxiliary path.",
        ha='center', va='bottom', fontsize=8, color="#888")

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=DPI)
plt.close(fig)
print(f"Wrote {OUT_PDF} ({os.path.getsize(OUT_PDF)}B) and {OUT_PNG}")