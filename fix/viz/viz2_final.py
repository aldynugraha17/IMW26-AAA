"""
Visualization 2: deadline sweep (T = 192..249) for the 25-task RCPSP-TCT instance.
Data below is the result of the CP-SAT two-stage exhaustive sweep:
  stage 1: min crash cost Z* s.t. Cmax <= T
  stage 2: enumerate ALL distinct optimal crash vectors (no-good cuts)
Produces 3 variants: stacked scatter (recommended), dual-axis, lollipop. PNG + PDF.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------- output folder ----------------
from pathlib import Path
current_dir = Path(__file__).resolve().parent

OUT = current_dir/ "figure"   # sesuaikan
# OUT = r'E:\p2ms\figures'   # sesuaikan
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUT, f'{name}.png')
    fig.savefig(p, dpi=200)
    fig.savefig(os.path.join(OUT, f'{name}.pdf'))   # vector, best for beamer
    plt.close(fig)
    print('saved ->', p)


# ---------------- sweep data (CP-SAT exhaustive enumeration) ----------------
Ts = list(range(192, 250))
Z = [6580, 6330, 6080, 5830, 5580, 5330, 5130, 4930, 4730, 4530,
     4330, 4150, 3970, 3790, 3610, 3430, 3250, 3070, 2890, 2710,
     2530, 2420, 2310, 2200, 2090, 1980, 1890, 1800, 1710, 1620,
     1530, 1450, 1370, 1290, 1210, 1130, 1050,  970,  890,  810,
      740,  670,  600,  540,  480,  420,  380,  340,  300,  260,
      220,  180,  150,  120,   90,   60,   30,    0]
N = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 2, 3, 4, 5, 5, 4, 3, 2, 1,
     1, 1, 1, 1, 1, 1, 2, 3, 3, 3,
     2, 1, 3, 5, 6, 5, 3, 1]
assert len(Ts) == len(Z) == len(N) == 58

HIGHLIGHT = ()   # case-study deadlines; kosongkan () jika tak ingin highlight
BLUE, ORANGE, HL = '#1f4e79', '#e07b39', '#c0392b'

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False})


# ============ Variant A: stacked two panels, discrete scatter (recommended) ============
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 6.4), sharex=True,
    gridspec_kw={'height_ratios': [2, 1.2], 'hspace': 0.08})

# top: discrete scatter of Z*(T) at integer deadlines
ax1.scatter(Ts, Z, s=26, color=BLUE, zorder=3)
for t in HIGHLIGHT:
    i = Ts.index(t)
    ax1.scatter([t], [Z[i]], s=55, color=HL, zorder=4)
    ax1.annotate(f'$T{{=}}{t}$: $Z^*{{=}}{Z[i]}$', xy=(t, Z[i]),
                 xytext=(-8, 14), textcoords='offset points',
                 ha='right', fontsize=9, color=HL, fontweight='bold')
ax1.set_ylabel('Minimum crashing cost $Z^*$')
ax1.set_title('Minimum crashing cost and number of optimal solutions '
              'per integer deadline (T = 192\u2013249)', fontsize=12)
ax1.set_ylim(-150, max(Z) * 1.06)
ax1.grid(alpha=0.3)

# bottom: bars of solution counts (discrete)
cols = [HL if t in HIGHLIGHT else ORANGE for t in Ts]
ax2.bar(Ts, N, width=0.7, color=cols, alpha=0.85)
for t in HIGHLIGHT:
    i = Ts.index(t)
    ax2.text(t, N[i] + 0.25, str(N[i]), ha='center',
             fontsize=9, color=HL, fontweight='bold')
ax2.set_ylabel('# optimal\nsolutions')
ax2.set_xlabel('Deadline $T$ (days)')
ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.set_ylim(0, max(N) + 1.5)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlim(min(Ts) - 1, max(Ts) + 1)
fig.tight_layout()
save(fig, 'viz2_stacked')


# ============ Variant B: dual axis (bars + scatter) ============
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax2.bar(Ts, N, width=0.7, color=ORANGE, alpha=0.55,
        label='Number of optimal solutions', zorder=2)
ax1.scatter(Ts, Z, s=26, color=BLUE, zorder=3,
            label='Minimum crashing cost $Z^*$')
for t in HIGHLIGHT:
    i = Ts.index(t)
    ax1.scatter([t], [Z[i]], s=55, color=HL, zorder=4)
    ax2.bar([t], [N[i]], width=0.7, color=HL, alpha=0.9, zorder=4)
    ax2.annotate(f'T={t}\n{N[i]} solutions', xy=(t, N[i]),
                 xytext=(t, N[i] + 0.9), ha='center', fontsize=9.5,
                 color=HL, fontweight='bold',
                 arrowprops=dict(arrowstyle='-', color=HL, lw=0.8))

ax1.set_xlabel('Deadline $T$ (days)')
ax1.set_ylabel('Minimum crashing cost $Z^*$', color=BLUE)
ax2.set_ylabel('Number of optimal solutions', color=ORANGE)
ax1.tick_params(axis='y', colors=BLUE)
ax2.tick_params(axis='y', colors=ORANGE)
ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.set_ylim(0, max(N) + 2.5)
ax1.set_ylim(-150, max(Z) * 1.06)
ax1.set_xlim(min(Ts) - 1, max(Ts) + 1)
ax1.set_title('Minimum crashing cost and number of optimal solutions '
              'across deadlines', fontsize=12)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False)
fig.tight_layout()
save(fig, 'viz2_dual_axis')


# ============ Variant C: lollipop (size-encoded frequency, scatter only) ============
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.scatter(Ts, Z, s=[28 * n for n in N],
            c=[HL if t in HIGHLIGHT else ORANGE for t in Ts],
            alpha=0.85, edgecolors='white', linewidths=0.7, zorder=3)
for i, t in enumerate(Ts):
    if N[i] > 1:
        ax1.annotate(str(N[i]), xy=(t, Z[i]), xytext=(0, 11),
                     textcoords='offset points', ha='center', fontsize=8.5,
                     color=HL if t in HIGHLIGHT else '#8a4a1f',
                     fontweight='bold')
ax1.set_xlabel('Deadline $T$ (days)')
ax1.set_ylabel('Minimum crashing cost $Z^*$')
ax1.set_title('Time\u2013cost frontier; marker size/label = number of optimal '
              'solutions per deadline', fontsize=12)
ax1.grid(alpha=0.3)
fig.tight_layout()
save(fig, 'viz2_lollipop')

print('working dir:', os.getcwd())
print('output dir :', OUT)
