# -*- coding: utf-8 -*-
"""Visualisasi Gantt chart hasil SOAC, bergaya solver_base.py milik adiel:
dua panel berdampingan (Baseline vs Optimized), bar biru = normal,
merah = crashed, garis putus-putus merah = deadline / project end date."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_gantt_comparison(problem, d_opt, output_path, title_suffix=""):
    """Gambar Gantt baseline (d = d_max) vs jadwal hasil crashing d_opt.

    Parameters
    ----------
    problem : ProjectCrashingProblem
    d_opt : array-like of int
        Vektor durasi hasil SOAC (satu solusi).
    output_path : str
        Path file PNG keluaran.
    """
    d_opt = np.asarray(d_opt, dtype=int)
    d_base = problem.d_max
    s_b, e_b = problem.schedule(d_base)
    s_o, e_o = problem.schedule(d_opt)
    names = problem.task_names
    n = problem.n_tasks

    order_b = sorted(range(n), key=lambda j: (s_b[j], e_b[j], names[j]))
    order_o = sorted(range(n), key=lambda j: (s_o[j], e_o[j], names[j]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, max(8, 0.45 * n)), sharey=False)

    def draw(ax, order, s, e, colors, title, end_line=None):
        for idx, j in enumerate(order):
            dur = int(e[j] - s[j])
            ax.barh(idx, dur, left=s[j], height=0.6, color=colors[j],
                    edgecolor="black", linewidth=0.5)
            label = str(dur)
            if dur > 2:
                ax.text(s[j] + dur / 2, idx, label, va="center", ha="center",
                        color="white", fontsize=8, weight="bold")
            else:
                ax.text(e[j] + 0.5, idx, label, va="center", ha="left",
                        color="black", fontsize=8)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel("Project Day", fontsize=12)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[j] for j in order], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        if end_line is not None:
            ax.axvline(x=end_line, color="red", linestyle="--", linewidth=1.5)

    # Panel kiri: baseline (semua biru)
    draw(ax1, order_b, s_b, e_b, ["#3498db"] * n,
         f"Original Schedule (Baseline) — makespan {e_b.max()}")

    # Panel kanan: optimized (merah jika di-crash)
    crashed = d_opt < d_base
    colors_o = ["#e74c3c" if crashed[j] else "#3498db" for j in range(n)]
    z = problem.crash_cost(d_opt)
    draw(ax2, order_o, s_o, e_o, colors_o,
         f"SOAC Optimized{title_suffix} — makespan {e_o.max()}, "
         f"crash cost = {z:.0f}",
         end_line=e_o.max())

    legend = [
        Patch(facecolor="#3498db", edgecolor="black", label="Normal (No Crash)"),
        Patch(facecolor="#e74c3c", edgecolor="black",
              label="Crashed (durasi < normal)"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5,
               label=f"Project End Date (Day {e_o.max()}) | "
                     f"Deadline T = {problem.deadline}"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
               frameon=True)
    fig.suptitle(f"Project Crashing via SOAC — {problem.name}", fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path




def plot_solution_frequency(problem, roots, output_path):
    """Duplikat gaya plot_makespan_distribution.py adiel: bar frekuensi
    makespan seluruh solusi optimum yang ditemukan, dengan Z di judul.
    Panel kanan: frekuensi tiap task di-crash pada himpunan solusi."""
    from collections import Counter

    roots = np.atleast_2d(roots).astype(int)
    n_sol = len(roots)
    z = problem.crash_cost(roots[0])
    makespans = [problem.makespan(d) for d in roots]
    counts = Counter(makespans)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # --- Panel kiri: distribusi makespan (ala adiel) ---
    x_vals = sorted(counts)
    y_vals = [counts[x] for x in x_vals]
    bars = ax1.bar([str(x) for x in x_vals], y_vals, color="#3498db",
                   width=0.6, edgecolor="none", zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05, f"{int(h)}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color="#2c3e50")
    ax1.set_title(f"Distribution of Makespans for SOAC Optimal Solutions\n"
                  f"(Crash Cost Z = {z:.1f}, {n_sol} solusi)",
                  fontsize=13, fontweight="bold", pad=15, color="#2c3e50")
    ax1.set_xlabel("Project Makespan (Days)", fontsize=11, fontweight="bold",
                   labelpad=10, color="#2c3e50")
    ax1.set_ylabel("Number of Solutions", fontsize=11, fontweight="bold",
                   labelpad=10, color="#2c3e50")
    ax1.set_ylim(0, max(y_vals) + 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    # --- Panel kanan: frekuensi task di-crash antar solusi ---
    y_mat = problem.d_max[None, :] - roots            # (n_sol, n_tasks)
    crashed_freq = (y_mat > 0).sum(axis=0)
    active = np.where(crashed_freq > 0)[0]
    labels = [problem.task_names[j] for j in active]
    bars2 = ax2.barh(range(len(active)), crashed_freq[active],
                     color="#e74c3c", height=0.55, zorder=3)
    for i, j in enumerate(active):
        mean_y = y_mat[:, j].mean()
        ax2.text(crashed_freq[j] + 0.05, i,
                 f"{crashed_freq[j]}/{n_sol} (rata2 y={mean_y:.1f})",
                 va="center", fontsize=9, color="#2c3e50")
    ax2.set_yticks(range(len(active)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, n_sol + 1.2)
    ax2.set_title(f"Crash Frequency per Task across {n_sol} Optimal Solutions",
                  fontsize=13, fontweight="bold", pad=15, color="#2c3e50")
    ax2.set_xlabel("Number of Solutions where Task is Crashed",
                   fontsize=11, fontweight="bold", labelpad=10, color="#2c3e50")
    ax2.grid(axis="x", linestyle="--", alpha=0.7, zorder=0)

    fig.suptitle(f"{problem.name} | Deadline T = {problem.deadline}",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    import numpy as np
    from pysne.solver import solve_system
    from run_crashing_adiel import load_tasks, DATA, DEADLINE
    from project_crashing_problem import ProjectCrashingProblem

    problem = ProjectCrashingProblem(
        load_tasks(DATA), DEADLINE, unit_cube=True,
        params={"m_cluster": 32768 * 2, "k_cluster": 15, "gamma": 0.9,
                "r_cl": 0.95, "theta_cl": np.pi / 4,
                "sdoa_m": 1024, "sdoa_k_max": 1000,
                "sdoa_r": 0.97, "sdoa_theta": np.pi / 4,
                "delta": 0.00001, "epsilon": 1e-9},
    )
    result = solve_system(problem, problem.get_info()[1], verbose=True)
    roots = result["roots"]
    if len(roots) == 0:
        print("Tidak ada solusi feasible untuk digambar.")
    else:
        # Satu Gantt per solusi: gantt_sol001_ms242.png, gantt_sol002_ms242.png, ...
        for k, root in enumerate(roots):
            out = plot_gantt_comparison(
                problem, root,
                f"gantt_sol{k + 1:03d}_ms{problem.deadline}.png",
                title_suffix=f" (solusi #{k + 1} dari {len(roots)})")
            print("Gantt tersimpan di:", out)

        # Duplikat visualisasi frekuensi (gaya adiel) + Z di judul
        z = problem.crash_cost(roots[0])
        out = plot_solution_frequency(
            problem, roots,
            f"frequency_Z{z:.0f}_ms{problem.deadline}.png")
        print("Plot frekuensi tersimpan di:", out)