# -*- coding: utf-8 -*-
"""Debug runner SOAC project crashing.

Membungkus objek problem dengan proxy yang mencatat SEMUA pemanggilan method
oleh solve_system, tanpa perlu mengubah kode pysne:
  - jumlah call & waktu kumulatif per method (ketahuan hotspot-nya)
  - progres evaluasi fitness: n_eval, evals/detik, F terbaik sejauh ini
  - deteksi pergantian fase (clustering -> SDOA) dari pola batch size

Jalankan seperti run biasa. Untuk debug awal, pakai PARAMS_DEBUG dulu
(budget kecil) untuk memastikan pipeline jalan, baru naikkan ke PARAMS_FULL.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pysne.solver import solve_system
from project_crashing_problem_final_fix import ProjectCrashingProblem

# current_dir = Path(__file__).resolve().parent
# DATA = current_dir.parent / "adiel" / "data" / "activity_data_v3.json"
current_dir = Path(__file__).resolve().parent
DATA_DIR = current_dir.parent / "adiel" / "data"
DATA = DATA_DIR / "activity_data_v3.json"
RES_CAP = DATA_DIR / "resource_capacity_v3.json"
RES_REQ = DATA_DIR / "resource_requirements_v3.json"
DEADLINE = 245

# Budget penuh (punyamu sekarang) -- +-524rb eval fase cluster,
# +-307rb eval per cluster di fase SDOA.
PARAMS_FULL = {
    "m_cluster": 1024, "k_cluster": 300, "gamma": 0.99,
    "r_cl": 0.95, "theta_cl": np.pi / 2,
    "sdoa_m": 8096, "sdoa_k_max": 300,
    "sdoa_r": 0.97, "sdoa_theta": np.pi / 8,
    "delta": 0.4, "epsilon": 1e-9, "num_check_points": 2
}

# Budget kecil untuk verifikasi cepat (harusnya selesai < 1 menit).
PARAMS_DEBUG = {
    "m_cluster": 4096, "k_cluster": 100, "gamma": 0.95,
    "r_cl": 0.95, "theta_cl": np.pi / 4,
    "sdoa_m": 256, "sdoa_k_max": 60,
    "sdoa_r": 0.97, "sdoa_theta": np.pi / 4,
    "delta": 0.4, "epsilon": 1e-9,
}


def load_tasks(path):
    raw = json.load(open(path))
    return [{"name": k,
             "predecessors": v["required_activities"],
             "d_min": v["activity_min_time"],
             "d_max": v["activity_normal_time"],
             "crash_cost": v["crash_cost"]} for k, v in raw.items()]


class DebugProblem:
    """Proxy transparan di atas ProjectCrashingProblem.

    Semua atribut/method diteruskan ke objek asli; method callable dibungkus
    timer + counter. Progres dicetak tiap `report_every` evaluasi fitness
    atau tiap `report_sec` detik (mana yang tercapai duluan).
    """

    _OWN = ("_inner", "_stats", "_n_eval", "_n_call_eval", "_best_F",
            "_t0", "_t_last", "_report_every", "_report_sec",
            "_last_batch", "_phase")

    def __init__(self, inner, report_every=25000, report_sec=5.0):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_stats", defaultdict(lambda: [0, 0.0]))
        object.__setattr__(self, "_n_eval", 0)        # total titik dievaluasi
        object.__setattr__(self, "_n_call_eval", 0)   # total call fitness
        object.__setattr__(self, "_best_F", -np.inf)
        object.__setattr__(self, "_t0", time.perf_counter())
        object.__setattr__(self, "_t_last", time.perf_counter())
        object.__setattr__(self, "_report_every", report_every)
        object.__setattr__(self, "_report_sec", report_sec)
        object.__setattr__(self, "_last_batch", None)
        object.__setattr__(self, "_phase", "?")

    # ---- forwarding ----
    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        stats = self._stats

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = attr(*args, **kwargs)
            dt = time.perf_counter() - t0
            st = stats[name]
            st[0] += 1
            st[1] += dt
            self._maybe_track_fitness(name, out)
            return out

        return wrapped

    def __setattr__(self, name, value):
        if name in DebugProblem._OWN:
            object.__setattr__(self, name, value)
        else:  # kalau solve_system men-set atribut, teruskan ke objek asli
            setattr(self._inner, name, value)

    def __call__(self, *args, **kwargs):
        # kalau solve_system memanggil problem sebagai fungsi
        t0 = time.perf_counter()
        out = self._inner(*args, **kwargs)
        dt = time.perf_counter() - t0
        st = self._stats["__call__"]
        st[0] += 1
        st[1] += dt
        self._maybe_track_fitness("__call__", out)
        return out

    def __len__(self):
        return len(self._inner)

    # ---- pelacakan fitness ----
    def _maybe_track_fitness(self, name, out):
        """Anggap output numerik = nilai fitness; catat progres."""
        arr = None
        if isinstance(out, np.ndarray) and np.issubdtype(out.dtype, np.number):
            arr = out
        elif isinstance(out, (int, float, np.floating, np.integer)):
            arr = np.asarray([out], dtype=float)
        if arr is None or arr.size == 0:
            return

        n = int(arr.size)
        object.__setattr__(self, "_n_eval", self._n_eval + n)
        object.__setattr__(self, "_n_call_eval", self._n_call_eval + 1)
        try:
            m = float(np.nanmax(arr))
            if m > self._best_F:
                object.__setattr__(self, "_best_F", m)
        except (ValueError, TypeError):
            pass

        # deteksi pergantian fase dari perubahan ukuran batch
        if self._last_batch is not None and n != self._last_batch and n > 1:
            print(f"[fase?] ukuran batch berubah {self._last_batch} -> {n} "
                  f"(kemungkinan pindah fase clustering/SDOA atau cluster baru)")
            object.__setattr__(self, "_t_last", 0.0)  # paksa cetak progres
        object.__setattr__(self, "_last_batch", n)

        now = time.perf_counter()
        if (self._n_eval % self._report_every < n
                or now - self._t_last >= self._report_sec):
            el = now - self._t0
            rate = self._n_eval / el if el > 0 else 0.0
            print(f"[{el:8.1f}s] eval={self._n_eval:>10,} "
                  f"({rate:,.0f} eval/s) | call {name}() x{self._stats[name][0]:,} "
                  f"| F_best={self._best_F:.6f}")
            object.__setattr__(self, "_t_last", now)

    # ---- ringkasan akhir ----
    def print_summary(self):
        el = time.perf_counter() - self._t0
        print("\n================ RINGKASAN DEBUG ================")
        print(f"Total waktu     : {el:.1f}s")
        print(f"Total evaluasi  : {self._n_eval:,} titik "
              f"dalam {self._n_call_eval:,} call "
              f"({self._n_eval / el:,.0f} eval/s)")
        print(f"F terbaik       : {self._best_F:.6f}")
        print(f"{'method':<28}{'#call':>12}{'total(s)':>12}{'ms/call':>12}")
        for name, (cnt, tot) in sorted(self._stats.items(),
                                       key=lambda kv: -kv[1][1]):
            print(f"{name:<28}{cnt:>12,}{tot:>12.2f}"
                  f"{1000.0 * tot / cnt:>12.3f}")
        print("=================================================\n")


def main():
    params = PARAMS_FULL  # ganti ke PARAMS_FULL kalau pipeline sudah oke
    problem = ProjectCrashingProblem(load_tasks(DATA), DEADLINE,
                                     unit_cube=True, params=params)

    # estimasi budget di awal biar tahu harus nunggu berapa lama
    m, k = params["m_cluster"], params["k_cluster"]
    sm, sk = params["sdoa_m"], params["sdoa_k_max"]
    print(f"{problem.name} | makespan normal={problem.makespan(problem.d_max)}, "
          f"min={problem.makespan(problem.d_min)}")
    print(f"Perkiraan budget: fase cluster ~{m * (k + 1):,} eval, "
          f"fase SDOA ~{sm * sk:,} eval PER CLUSTER\n")
    print(f"param: {PARAMS_FULL}")


    dbg = DebugProblem(problem, report_every=25000, report_sec=5.0)

    t0 = time.time()
    result = solve_system(dbg, problem.get_info()[1], verbose=True)
    dt = time.time() - t0

    dbg.print_summary()

    roots = result["roots"]
    print(f"Waktu SOAC: {dt:.1f}s, cluster: {len(result['clusters'])}")
    if len(roots) == 0:
        print("SOAC tidak menemukan solusi feasible.")
        return
    costs = [problem.crash_cost(r) for r in roots]
    print(f"SOAC: {len(roots)} solusi, biaya terbaik = {min(costs)}")
    print(problem.report(roots[:3]))


if __name__ == "__main__":
    main()
