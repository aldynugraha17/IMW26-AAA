"""
Combined 2x2 Gantt figure for the T=241 scenario (beamer-ready).
Top-left : baseline (normal) schedule, makespan 249
Top-right: SPOC optimal solution 1 (Z*=260)
Bottom   : SPOC optimal solutions 2 and 3 (Z*=260)

Schedules are regenerated with a precedence + resource-constrained
earliest-start SGS, identical to the one used for the single-solution figures.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from pathlib import Path
current_dir = Path(__file__).resolve().parent

# DATA = Path('/mnt/project')
DATA = current_dir.parent
acts = json.load(open(DATA / 'activity_data_v3.json'))
cap  = json.load(open(DATA / 'resource_capacity_v3.json'))
req  = json.load(open(DATA / 'resource_requirements_v3.json'))
names = list(acts.keys())
T_DEADLINE = 241

# ---------------------------------------------------------------- SGS
def schedule(durations):
    horizon = sum(durations.values()) + 10
    usage = {r: [0] * horizon for r in cap}
    start, finish = {}, {}
    order, remaining = [], set(names)
    while remaining:
        for n in names:
            if n in remaining and all(p in order for p in acts[n]['required_activities']):
                order.append(n)
                remaining.discard(n)
    for n in order:
        d = durations[n]
        t = max([finish[p] for p in acts[n]['required_activities']], default=0)
        while True:
            ok = all(
                usage[r][day] + a <= cap[r]
                for r, a in req[n].items() if a
                for day in range(t, t + d)
            )
            if ok:
                break
            t += 1
        start[n], finish[n] = t, t + d
        for r, a in req[n].items():
            if a:
                for day in range(t, t + d):
                    usage[r][day] += a
    return start, finish

normal = {n: acts[n]['activity_normal_time'] for n in names}

# The three minimum-cost crash vectors found by SPOC at T=241 (Z* = 260)
solutions = [
    ('SPOC Optimal Solution 1 of 3', {'Site Work': 2, 'Insulation': 1,
                                      'Final Punch-out': 3, 'Cleaning': 2}),
    ('SPOC Optimal Solution 2 of 3', {'Site Work': 2, 'Insulation': 1,
                                      'Plumbing Trim': 2, 'Final Punch-out': 3}),
    ('SPOC Optimal Solution 3 of 3', {'Site Work': 2, 'Insulation': 1,
                                      'Plumbing Trim': 1, 'Final Punch-out': 3,
                                      'Cleaning': 1}),
]

base_start, base_finish = schedule(normal)
base_ms = max(base_finish.values())
# y-order: sorted by baseline start time (same as the original single figures)
y_order = sorted(names, key=lambda n: (base_start[n], names.index(n)))
y_pos = {n: len(y_order) - 1 - i for i, n in enumerate(y_order)}

panels = [('Baseline (Normal Schedule)', None, base_start, normal, base_ms, None)]
for label, cv in solutions:
    dur = {n: normal[n] - cv.get(n, 0) for n in names}
    s, f = schedule(dur)
    ms = max(f.values())
    z = sum(acts[n]['crash_cost'] * c for n, c in cv.items())
    panels.append((label, cv, s, dur, ms, z))

# ---------------------------------------------------------------- plot
plt.rcParams.update({'font.size': 11})
fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), sharex=True, sharey=True)
BLUE, RED = '#3B8FD4', '#D64541'

for ax, (label, cv, s, dur, ms, z) in zip(axes.flat, panels):
    for n in names:
        crashed = cv is not None and n in cv
        ax.barh(y_pos[n], dur[n], left=s[n], height=0.62,
                color=RED if crashed else BLUE, edgecolor='black', linewidth=0.5)
        d = dur[n]
        if d >= 4:
            ax.text(s[n] + d / 2, y_pos[n], str(d), ha='center', va='center',
                    color='white', fontsize=7.5, fontweight='bold')
        else:
            ax.text(s[n] + d + 1.5, y_pos[n], str(d), ha='left', va='center',
                    fontsize=7.5)
    if cv is not None:
        ax.axvline(T_DEADLINE, color='red', linestyle='--', linewidth=1.6)
        ax.set_title(f'{label}\nmakespan {ms}, crash cost $Z^*$ = {z:.0f}',
                     fontsize=12.5)
    else:
        ax.set_title(f'{label}\nmakespan {ms} (no crashing)', fontsize=12.5)
    ax.set_xlim(0, 262)
    ax.set_ylim(-0.8, len(names) - 0.2)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.tick_params(axis='y', labelsize=9)

for ax in axes[:, 0]:
    ax.set_yticks([y_pos[n] for n in y_order])
    ax.set_yticklabels(y_order)
for ax in axes[1, :]:
    ax.set_xlabel('Project Day', fontsize=12)

fig.suptitle(f'Project Crashing via SPOC — RCPSP–TCT (n = 25, T = {T_DEADLINE})',
             fontsize=16, y=0.985)
handles = [
    mpatches.Patch(facecolor=BLUE, edgecolor='black', label='Normal (no crash)'),
    mpatches.Patch(facecolor=RED, edgecolor='black', label='Crashed (duration < normal)'),
    Line2D([0], [0], color='red', linestyle='--', linewidth=1.6,
           label=f'Deadline T = {T_DEADLINE}'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11.5,
           frameon=True, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.035, 1, 0.965])

out = current_dir/ "figure"
fig.savefig(out / 'gantt_combined_T241.png', dpi=300, bbox_inches='tight')
fig.savefig(out / 'gantt_combined_T241.pdf', bbox_inches='tight')
print('saved.')
