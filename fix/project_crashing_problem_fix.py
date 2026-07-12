# -*- coding: utf-8 -*-
"""
ProjectCrashingProblem: adaptasi SPOC (Sidarto-Kania-Sumarti 2017 + SPOC
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

Transformasi ke bentuk maksimisasi SPOC (fitness F dalam (0, 1]):

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
    def __init__(self, tasks, deadline, params=None, unit_cube=True,
                 cost_tolerance=0.0, resource_capacity=None,
                 resource_requirements=None):
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

        self.Z_max = float(np.sum(self.c * (self.d_max - self.d_min)))
        if self.Z_max <= 0:
            self.Z_max = 1.0

        name_to_idx = {t["name"]: i for i, t in enumerate(self.tasks)}
        self.pred_idx = [[name_to_idx[p] for p in t.get("predecessors", [])]
                         for t in self.tasks]
        self.succ_idx = [[] for _ in range(self.n_tasks)]
        for j, preds in enumerate(self.pred_idx):
            for p in preds:
                self.succ_idx[p].append(j)
        self.topo_order = self._topological_order()

        self.integer_domain = [(int(lo), int(hi))
                               for lo, hi in zip(self.d_min, self.d_max)]
        self._widened = create_continuous_bounds(self.integer_domain, margin=0.5)

        # ---------- Sumber daya (opsional) ----------
        self.has_resources = (resource_capacity is not None
                              and resource_requirements is not None)
        if self.has_resources:
            self.resource_names = list(resource_capacity.keys())
            self.Cap = np.array([int(resource_capacity[r])
                                 for r in self.resource_names])
            self.Req = np.array(
                [[int(resource_requirements.get(nm, {}).get(r, 0))
                  for r in self.resource_names]
                 for nm in self.task_names])            # (n_tasks, n_res)
            self._need = [np.where(self.Req[j] > 0)[0] for j in range(self.n_tasks)]
            self._H = int(self.d_max.sum()) + 2          # horizon aman utk SGS
            self._sched_cache = {}                       # memo: tuple(d) -> (mk,s,e)
        # --------------------------------------------

        self._params = dict(params) if params else {}
        super().__init__()
        self.equations = None

    @property
    def name(self):
        tag = "RCPSP" if self.has_resources else "CPM"
        return f"ProjectCrashing[{tag}](n={self.n_tasks}, T={self.deadline})"

    @property
    def optima_type(self):
        return "max"

    def get_info(self):
        if self.unit_cube:
            domain = [(0.0, 1.0)] * self.n_tasks
        else:
            domain = self._widened
        default_params = {
            "m_cluster": 32768, "k_cluster": 12, "gamma": 0.85,
            "r_cl": 0.95, "theta_cl": np.pi / 4, "num_check_points": 1,
            "sdoa_m": 1024, "sdoa_k_max": 300,
            "sdoa_r": 0.97, "sdoa_theta": np.pi / 4,
            "delta": 0.4, "epsilon": 1e-9,
        }
        default_params.update(self._params)
        return domain, default_params

    # ---------------- decode ----------------
    def decode(self, x):
        x = np.asarray(x, dtype=float)
        if self.unit_cube:
            lo = self.d_min - 0.5; hi = self.d_max + 0.5
            scaled = lo + np.clip(x, 0.0, 1.0) * (hi - lo)
        else:
            scaled = x
        d = np.rint(scaled).astype(int)
        return np.clip(d, self.d_min, self.d_max)

    def decode_batch(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.unit_cube:
            lo = self.d_min - 0.5; hi = self.d_max + 0.5
            scaled = lo + np.clip(X, 0.0, 1.0) * (hi - lo)
        else:
            scaled = X
        D = np.rint(scaled).astype(int)
        return np.clip(D, self.d_min, self.d_max)

    # ------------- penjadwalan CPM (tanpa sumber daya) -------------
    def cpm_schedule(self, d):
        s = np.zeros(self.n_tasks, dtype=int); e = np.zeros(self.n_tasks, dtype=int)
        for j in self.topo_order:
            s[j] = max((e[p] for p in self.pred_idx[j]), default=0)
            e[j] = s[j] + int(d[j])
        return s, e

    def makespan(self, d):
        _, e = self.cpm_schedule(d)
        return int(e.max()) if self.n_tasks else 0

    def _forward_pass_batch(self, D):
        B = D.shape[0]
        E = np.zeros((B, self.n_tasks), dtype=D.dtype)
        for j in self.topo_order:
            preds = self.pred_idx[j]
            if preds:
                s = E[:, preds[0]]
                for p in preds[1:]:
                    s = np.maximum(s, E[:, p])
            else:
                s = np.zeros(B, dtype=D.dtype)
            E[:, j] = s + D[:, j]
        return E

    # ------------- penjadwalan resource-feasible (SGS) -------------
    def _lft_order(self, d):
        s, e = self.cpm_schedule(d)
        Tref = max(int(e.max()), self.deadline)
        LF = np.full(self.n_tasks, Tref, int)
        for j in reversed(self.topo_order):
            ss = [LF[k] - int(d[k]) for k in self.succ_idx[j]]
            LF[j] = min(ss) if ss else Tref
        return sorted(range(self.n_tasks), key=lambda j: (LF[j], s[j], j))

    def _serial_sgs(self, d, order):
        usage = np.zeros((len(self.resource_names), self._H), int)
        s = np.zeros(self.n_tasks, int); e = np.zeros(self.n_tasks, int)
        for j in order:
            est = max((e[p] for p in self.pred_idx[j]), default=0)
            dj = int(d[j]); need = self._need[j]; t = est
            if need.size == 0:
                s[j] = est; e[j] = est + dj; continue
            capN = self.Cap[need][:, None]; rN = self.Req[j, need][:, None]
            while True:
                if bool((usage[need, t:t + dj] + rN <= capN).all()):
                    for r in need:
                        usage[r, t:t + dj] += self.Req[j, r]
                    s[j] = t; e[j] = t + dj; break
                t += 1
        return int(e.max()), s, e

    def _parallel_sgs(self, d):
        usage = np.zeros((len(self.resource_names), self._H), int)
        s = -np.ones(self.n_tasks, int); e = -np.ones(self.n_tasks, int)
        remaining = set(range(self.n_tasks))
        while remaining:
            elig = [j for j in remaining if all(e[p] >= 0 for p in self.pred_idx[j])]
            elig.sort(key=lambda j: (max((e[p] for p in self.pred_idx[j]), default=0), j))
            j = elig[0]
            est = max((e[p] for p in self.pred_idx[j]), default=0)
            dj = int(d[j]); need = self._need[j]; t = est
            if need.size == 0:
                s[j] = est; e[j] = est + dj
            else:
                capN = self.Cap[need][:, None]; rN = self.Req[j, need][:, None]
                while True:
                    if bool((usage[need, t:t + dj] + rN <= capN).all()):
                        for r in need:
                            usage[r, t:t + dj] += self.Req[j, r]
                        s[j] = t; e[j] = t + dj; break
                    t += 1
            remaining.discard(j)
        return int(e.max()), s, e

    def _resource_schedule(self, d):
        key = tuple(int(x) for x in d)
        cached = self._sched_cache.get(key)
        if cached is not None:
            return cached
        cpm_ms = self.makespan(d)                       # batas bawah makespan
        m1, s1, e1 = self._serial_sgs(d, self._lft_order(d))
        if m1 <= cpm_ms:                                # capai batas bawah -> optimal
            res = (m1, s1, e1)
        else:                                           # baru pakai parallel utk perketat
            m2, s2, e2 = self._parallel_sgs(d)
            res = (m1, s1, e1) if m1 <= m2 else (m2, s2, e2)
        self._sched_cache[key] = res
        return res

    def resource_makespan(self, d):
        if not self.has_resources:
            return self.makespan(d)
        return self._resource_schedule(np.asarray(d))[0]

    def schedule(self, d):
        """Jadwal untuk pelaporan: resource-feasible bila ada sumber daya."""
        if self.has_resources:
            _, s, e = self._resource_schedule(np.asarray(d))
            return s, e
        return self.cpm_schedule(d)

    def crash_cost(self, d):
        return float(np.sum(self.c * (self.d_max - np.asarray(d))))

    # ---------------- fitness ----------------
    def g_func(self, x):
        return self.evaluate_fitness(x)

    def evaluate_fitness(self, x):
        x = np.asarray(x, dtype=float)
        single = (x.ndim == 1)
        X = x.reshape(1, -1) if single else x

        D = self.decode_batch(X)
        E_cpm = self._forward_pass_batch(D)
        cpm_ms = E_cpm.max(axis=1)
        Z_norm = ((self.d_max - D) @ self.c) / self.Z_max

        if self.has_resources:
            # makespan efektif: CPM (batas bawah) di daerah jauh-infeasible,
            # SGS resource-feasible hanya untuk yang lolos CPM (hemat komputasi).
            eff = cpm_ms.astype(float).copy()
            near = np.where(cpm_ms <= self.deadline)[0]
            for i in near:
                eff[i] = self.resource_makespan(D[i])
            makespan = eff
        else:
            makespan = cpm_ms

        violation = np.maximum(0, makespan - self.deadline)
        F = np.where(violation == 0,
                     1.0 / (1.0 + Z_norm),
                     1.0 / (2.0 + Z_norm + violation))
        return float(F[0]) if single else F

    # ------------- seleksi akhir (feasibility resource-aware) -------------
    def select_final_roots(self, candidates):
        evaluated = {}
        for cand in candidates:
            d = self.decode(cand)
            mk = self.resource_makespan(d) if self.has_resources else self.makespan(d)
            if mk > self.deadline:
                continue
            evaluated.setdefault(tuple(d.tolist()), self.crash_cost(d))
        if not evaluated:
            return np.array([])
        best_cost = min(evaluated.values())
        tol = max(self.cost_tolerance, 1e-9 * max(1.0, abs(best_cost)))
        keep = [np.array(d, dtype=float)
                for d, z in sorted(evaluated.items(), key=lambda kv: kv[1])
                if z <= best_cost + tol]
        return np.array(keep)

    def select_final_optimal(self, candidates):
        return self.select_final_roots(candidates)

    # ---------------- pelaporan ----------------
    def report(self, roots):
        lines = []
        for k, root in enumerate(np.atleast_2d(roots) if len(roots) else []):
            d = np.asarray(root, dtype=int)
            s, e = self.schedule(d)
            y = self.d_max - d
            tag = " (resource-feasible)" if self.has_resources else ""
            lines.append(f"Solusi #{k + 1}: Z = {self.crash_cost(d):.2f}, "
                         f"makespan = {e.max()} (deadline {self.deadline}){tag}")
            for j in range(self.n_tasks):
                lines.append(
                    f"  {self.task_names[j]:<22} d={d[j]:>3} "
                    f"(normal {self.d_max[j]}, min {self.d_min[j]}) "
                    f"crash y={y[j]}  s={s[j]:>3}  e={e[j]:>3}")
        return "\n".join(lines) if lines else "Tidak ada solusi feasible."

    def _topological_order(self):
        indeg = [len(p) for p in self.pred_idx]
        queue = [j for j in range(self.n_tasks) if indeg[j] == 0]
        order = []
        while queue:
            u = queue.pop()
            order.append(u)
            for v in self.succ_idx[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if len(order) != self.n_tasks:
            raise ValueError("Precedence mengandung siklus.")
        return order

    @classmethod
    def from_json(cls, path, deadline, resource_capacity_path=None,
                  resource_requirements_path=None, **kwargs):
        with open(path) as f:
            data = json.load(f)
        tasks = data["tasks"] if isinstance(data, dict) else data
        rc = rr = None
        if resource_capacity_path:
            rc = json.load(open(resource_capacity_path))
        if resource_requirements_path:
            rr = json.load(open(resource_requirements_path))
        return cls(tasks, deadline, resource_capacity=rc,
                   resource_requirements=rr, **kwargs)


