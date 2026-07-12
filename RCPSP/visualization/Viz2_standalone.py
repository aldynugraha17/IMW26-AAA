"""
Visualization 2: deadline sweep (T = 192..249) for the 25-task RCPSP-TCT instance.
Data below is the ground-truth result of CP-SAT two-stage sweep:
  stage 1: min crash cost Z* s.t. Cmax <= T
  stage 2: enumerate ALL distinct optimal crash vectors (no-good cuts)
Produces 3 variants: stacked (recommended), dual-axis, lollipop. PNG + PDF.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------- ground-truth sweep data ----------------
Ts = list(range(232, 250))
# Z = [6580, 6330, 6080, 5830, 5580, 5330, 5130, 4930, 4730, 4530,
#      4330, 4150, 3970, 3790, 3610, 3430, 3250, 3070, 2890, 2710,
#      2530, 2420, 2310, 2200, 2090, 1980, 1890, 1800, 1710, 1620,
#      1530, 1450, 1370, 1290, 1210, 1130, 1050,  970,  890,  810,
Z = [740,  670,  600,  540,  480,  420,  380,  340,  300,  260,
      220,  180,  150,  120,   90,   60,   30,    0]
# N = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#      1, 2, 3, 4, 5, 5, 4, 3, 2, 1,
N = [1, 1, 1, 1, 1, 1, 2, 3, 3, 3,
     2, 1, 3, 5, 6, 5, 3, 1]
assert len(Ts) == len(Z) == len(N) == 18

HIGHLIGHT = ()          # case-study deadlines
BLUE, ORANGE, HL = '#1f4e79', '#e07b39', '#c0392b'

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False})


def save(fig, name):
    fig.savefig(f'{name}.png', dpi=200)
    fig.savefig(f'{name}.pdf')          # vector, best for beamer
    plt.close(fig)


# ============ Variant A: stacked two panels (recommended) ============
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 6.4), sharex=True,
    gridspec_kw={'height_ratios': [2, 1.2], 'hspace': 0.08})

ax1.plot(Ts, Z, drawstyle='steps-post', color=BLUE, lw=2.2)
ax1.fill_between(Ts, Z, step='post', color=BLUE, alpha=0.08)
ax1.set_ylabel('Minimum crashing cost $Z^*$')
ax1.set_title('Time–cost frontier and multiplicity of optimal solutions '
              '(T = 232–249)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

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


# ============ Variant B: dual axis (bars + step line) ============
fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax2.bar(Ts, N, width=0.7, color=ORANGE, alpha=0.55,
        label='Number of optimal solutions', zorder=2)
ax1.plot(Ts, Z, drawstyle='steps-post', color=BLUE, lw=2.2,
         label='Minimum crashing cost $Z^*$', zorder=3)

for t in HIGHLIGHT:
    i = Ts.index(t)
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
ax1.set_xlim(min(Ts) - 1, max(Ts) + 1)
ax1.set_title('Minimum crashing cost and multiplicity of optimal solutions '
              'across deadlines', fontsize=12)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False)
fig.tight_layout()
save(fig, 'viz2_dual_axis')


# ============ Variant C: lollipop (size-encoded frequency on cost curve) ============
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(Ts, Z, drawstyle='steps-post', color=BLUE, lw=1.8, alpha=0.8, zorder=2)
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
ax1.set_title('Time–cost frontier; marker size/label = number of optimal '
              'solutions per deadline', fontsize=12)
ax1.grid(alpha=0.3)
fig.tight_layout()
save(fig, 'viz2_lollipop')

print('saved: viz2_stacked, viz2_dual_axis, viz2_lollipop (.png + .pdf)')

import os
OUT = r'E:\p2ms\figures'   # sesuaikan
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=200)
    fig.savefig(os.path.join(OUT, f'{name}.pdf'))
    plt.close(fig)

import os
print(os.getcwd())

