# -*- coding: utf-8 -*-
"""
ProjectCrashingProblem: adaptasi SOAC (Sidarto-Kania-Sumarti 2017 + SOAC
Diophantine 2023) untuk Project Crashing dengan solusi integer,
terintegrasi penuh dengan pipeline pysne (branch eksperimen).

Formulasi
---------
Variabel keputusan : d = (d_1, ..., d_n), durasi integer tiap task,
                     d_min_j <= d_j <= d_max_j.
Variabel turunan   : s_j = max_{p in pred(j)} e_p  (0 jika tanpa predecessor)
                     e_j = s_j + d_j                (forward pass / CPM)
                     y_j = d_max_j - d_j            (jumlah hari crash)
Objektif           : minimize  Z(d) = sum_j c_j * y_j
Kendala            : makespan(d) = max_j e_j <= T_deadline

Transformasi ke bentuk maksimisasi SOAC (fitness F dalam (0, 1]):

    Z_norm  = Z / Z_max,  Z_max = sum_j c_j (d_max_j - d_min_j)
    v       = max(0, makespan - T_deadline)          (pelanggaran deadline)

    F(d) = 1 / (1 + Z_norm)              jika feasible  -> F in [1/2, 1]
    F(d) = 1 / (2 + Z_norm + v)          jika infeasible -> F <  1/2

Sehingga setiap solusi feasible selalu lebih baik daripada solusi
infeasible mana pun, dan di dalam wilayah infeasible masih ada gradien
menuju feasibility (v mengecil -> F membesar). Semua vektor durasi dengan
biaya crash minimum yang sama membentuk plateau F maksimum -- struktur
multimodal yang persis sama dengan "banyak akar" pada kasus Diophantine.

Representasi titik (prosedur integer ala paper Diophantine):
  Titik spiral x bergerak di ruang kontinu. Setiap dimensi j didekode:
    - unit_cube=True : x_j in [0,1] dipetakan affine ke
                       [d_min_j - 0.5, d_max_j + 0.5], lalu dibulatkan.
    - unit_cube=False: domain kontinu langsung [d_min_j - 0.5, d_max_j + 0.5]
                       (persis pola DiophantineProblem di pysne).
  Hasil pembulatan di-clamp ke [d_min_j, d_max_j] agar titik tepat di tepi
  domain yang melebar tidak jatuh keluar rentang integer.
"""

import json
import numpy as np

from pysne.problems.base import BaseProblem
from pysne.utils import create_continuous_bounds


class ProjectCrashingProblem(BaseProblem):
    """Problem project crashing (integer) untuk solver SOAC pysne."""

    problem_type = "Diophantine"  # evaluasi diskret via pembulatan, ala SOAC-Diophantine

    def __init__(self, tasks, deadline, params=None, unit_cube=True,
                 cost_tolerance=0.0):
        """
        Parameters
        ----------
        tasks : list of dict
            Tiap task: {
              "name": str,
              "predecessors": [nama task pendahulu],
              "d_min": int,   # durasi minimum setelah crashing
              "d_max": int,   # durasi normal
              "crash_cost": float  # c_j, biaya per hari crashing
            }
        deadline : int
            T_deadline, batas hari selesai proyek (makespan <= deadline).
        params : dict, optional
            Hyperparameter SOAC (m_cluster, k_cluster, gamma, r_cl, theta_cl,
            sdoa_m, sdoa_k_max, sdoa_r, sdoa_theta, delta, epsilon).
        unit_cube : bool
            True  -> spiral berotasi di [0,1]^n, dekode affine di fitness
                     (sesuai ide domain [0,1]^#task).
            False -> spiral langsung di [d_min-0.5, d_max+0.5]^n
                     (pola DiophantineProblem pysne). Keduanya ekuivalen
                     secara sampling Sobol; beda hanya geometri rotasi.
        cost_tolerance : float
            Toleransi biaya pada seleksi akhir: kandidat dengan
            Z <= Z_best + cost_tolerance ikut dilaporkan. 0.0 = hanya
            biaya minimum eksak.
        """
        self.tasks = list(tasks)
        self.task_names = [t["name"] for t in self.tasks]
        self.n_tasks = len(self.tasks)
        self.deadline = int(deadline)
        self.unit_cube = bool(unit_cube)
        self.cost_tolerance = float(cost_tolerance)

        self.d_min = np.array([int(t["d_min"]) for t in self.tasks])
        self.d_max = np.array([int(t["d_max"]) for t in self.tasks])
        self.c = np.array([float(t["crash_cost"]) for t in self.tasks])
        if np.any(self.d_min > self.d_max):
            raise ValueError("Ada task dengan d_min > d_max.")

        # Z_max untuk normalisasi (hindari 0 jika tak ada yang bisa di-crash)
        self.Z_max = float(np.sum(self.c * (self.d_max - self.d_min)))
        if self.Z_max <= 0:
            self.Z_max = 1.0

        # Precedence: indeks predecessor + urutan topologis (sekali saja)
        name_to_idx = {t["name"]: i for i, t in enumerate(self.tasks)}
        self.pred_idx = [
            [name_to_idx[p] for p in t.get("predecessors", [])]
            for t in self.tasks
        ]
        self.topo_order = self._topological_order()

        # Domain integer dan domain kontinu yang dilebarkan +-0.5
        self.integer_domain = [(int(lo), int(hi))
                               for lo, hi in zip(self.d_min, self.d_max)]
        self._widened = create_continuous_bounds(self.integer_domain, margin=0.5)

        self._params = dict(params) if params else {}
        super().__init__()  # BaseProblem membaca get_info() -> set self.domain, n_var
        self.equations = None  # bukan SNE; matikan early-stopping residual di SDOA

    # ------------------------------------------------------------------ #
    # Wajib dari BaseProblem
    # ------------------------------------------------------------------ #
    @property
    def name(self):
        return f"ProjectCrashing(n={self.n_tasks}, T={self.deadline})"

    @property
    def optima_type(self):
        return "max"  # F sudah bentuk maksimisasi

    def get_info(self):
        if self.unit_cube:
            domain = [(0.0, 1.0)] * self.n_tasks
        else:
            domain = self._widened
        default_params = {
            "m_cluster": 256, "k_cluster": 12, "gamma": 0.85,
            "r_cl": 0.95, "theta_cl": np.pi / 4, "num_check_points": 1,
            "sdoa_m": 64, "sdoa_k_max": 120,
            "sdoa_r": 0.95, "sdoa_theta": np.pi / 4,
            "delta": 0.4, "epsilon": 1e-7,
        }
        default_params.update(self._params)
        return domain, default_params

    # ------------------------------------------------------------------ #
    # Dekode titik kontinu -> vektor durasi integer
    # ------------------------------------------------------------------ #
    def decode(self, x):
        """x (kontinu) -> d (durasi integer, sudah di-clamp ke [d_min, d_max])."""
        x = np.asarray(x, dtype=float)
        if self.unit_cube:
            lo = self.d_min - 0.5
            hi = self.d_max + 0.5
            scaled = lo + np.clip(x, 0.0, 1.0) * (hi - lo)  # [0,1] -> [dmin-0.5, dmax+0.5]
        else:
            scaled = x
        d = np.rint(scaled).astype(int)          # pembulatan ke integer terdekat
        return np.clip(d, self.d_min, self.d_max)

    # ------------------------------------------------------------------ #
    # Penjadwalan (forward pass) dan biaya
    # ------------------------------------------------------------------ #
    def schedule(self, d):
        """Isi vektor s dan e via forward pass mengikuti urutan topologis."""
        s = np.zeros(self.n_tasks, dtype=int)
        e = np.zeros(self.n_tasks, dtype=int)
        for j in self.topo_order:
            s[j] = max((e[p] for p in self.pred_idx[j]), default=0)
            e[j] = s[j] + int(d[j])
        return s, e

    def crash_cost(self, d):
        """Z(d) = sum_j c_j * (d_max_j - d_j)."""
        return float(np.sum(self.c * (self.d_max - np.asarray(d))))

    def makespan(self, d):
        _, e = self.schedule(d)
        return int(e.max()) if self.n_tasks else 0

    # ------------------------------------------------------------------ #
    # Fitness F(x) yang dipakai clustering & SDOA
    # ------------------------------------------------------------------ #
    def g_func(self, x):
        return self.evaluate_fitness(x)

    def evaluate_fitness(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:  # dukung evaluasi batch dari engine SDOA
            return np.array([self.evaluate_fitness(row) for row in x])

        d = self.decode(x)
        Z_norm = self.crash_cost(d) / self.Z_max
        violation = max(0, self.makespan(d) - self.deadline)
        if violation == 0:
            return 1.0 / (1.0 + Z_norm)            # feasible: F dalam [1/2, 1]
        return 1.0 / (2.0 + Z_norm + violation)    # infeasible: F < 1/2

    # ------------------------------------------------------------------ #
    # Seleksi akhir: kumpulkan SEMUA vektor durasi feasible berbiaya minimum
    # ------------------------------------------------------------------ #
    def select_final_roots(self, candidates):
        evaluated = {}
        for cand in candidates:
            d = self.decode(cand)
            if self.makespan(d) > self.deadline:
                continue
            evaluated.setdefault(tuple(d.tolist()), self.crash_cost(d))

        if not evaluated:
            return np.array([])

        best_cost = min(evaluated.values())
        keep = [np.array(d, dtype=float)
                for d, z in sorted(evaluated.items(), key=lambda kv: kv[1])
                if z <= best_cost + self.cost_tolerance]
        return np.array(keep)

    def select_final_optimal(self, candidates):
        return self.select_final_roots(candidates)

    # ------------------------------------------------------------------ #
    # Pelaporan
    # ------------------------------------------------------------------ #
    def report(self, roots):
        """Cetak solusi: durasi, hari crash y_j, jadwal, biaya, makespan."""
        lines = []
        for k, root in enumerate(np.atleast_2d(roots) if len(roots) else []):
            d = np.asarray(root, dtype=int)
            s, e = self.schedule(d)
            y = self.d_max - d
            lines.append(f"Solusi #{k + 1}: Z = {self.crash_cost(d):.2f}, "
                         f"makespan = {e.max()} (deadline {self.deadline})")
            for j in range(self.n_tasks):
                lines.append(
                    f"  {self.task_names[j]:<22} d={d[j]:>3} "
                    f"(normal {self.d_max[j]}, min {self.d_min[j]}) "
                    f"crash y={y[j]}  s={s[j]:>3}  e={e[j]:>3}"
                )
        return "\n".join(lines) if lines else "Tidak ada solusi feasible."

    # ------------------------------------------------------------------ #
    # Utilitas
    # ------------------------------------------------------------------ #
    def _topological_order(self):
        indeg = [0] * self.n_tasks
        succ = [[] for _ in range(self.n_tasks)]
        for j, preds in enumerate(self.pred_idx):
            indeg[j] = len(preds)
            for p in preds:
                succ[p].append(j)
        queue = [j for j in range(self.n_tasks) if indeg[j] == 0]
        order = []
        while queue:
            u = queue.pop()
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if len(order) != self.n_tasks:
            raise ValueError("Precedence mengandung siklus.")
        return order

    @classmethod
    def from_json(cls, path, deadline, **kwargs):
        """Muat daftar task dari file JSON (list of dict)."""
        with open(path) as f:
            data = json.load(f)
        tasks = data["tasks"] if isinstance(data, dict) else data
        return cls(tasks, deadline, **kwargs)