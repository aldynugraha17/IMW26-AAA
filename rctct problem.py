"""
rctct_problem.py
================
Decoder + definisi problem RCTCT (Resource-Constrained Time-Cost Tradeoff)
untuk dioptimasi dengan pysne (SDOA + clustering, multimodal).

Skenario 1 Baseline (diskret linear, durasi INTEGER).

IDE INTI
--------
Domain pencarian SDOA TETAP berupa box statis. Yang "dinamis" (precedence +
resource) tidak masuk ke box domain, melainkan hidup di dalam decoder
(serial Schedule-Generation Scheme) di method `decode()`.

Vektor keputusan x dipecah dua blok:
  [ blok crash  |  blok priority ]
    - blok crash    : satu entri per aktivitas yang BISA di-crash (d_min < d_max).
                      nilainya c_i in [0, d_max_i - d_min_i]  (dibulatkan ke int).
                      durasi aktual d_i = d_max_i - c_i.
    - blok priority : satu entri per SEMUA aktivitas, key in [0,1], dipakai
                      memutus rebutan resource dalam serial-SGS.

`g_func` mengembalikan -total_cost karena pysne MEMAKSIMALKAN fitness,
sedangkan kita ingin MEMINIMALKAN biaya.
"""

import json
import numpy as np
from pysne.problems.base import MultimodalProblem
from pysne.utils import is_in_domain, filter_unique_roots


# --------------------------------------------------------------------------
# Loader data (menyesuaikan skema activity_data_v3.json milikmu)
# --------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_capacity(raw):
    """Terima {name: cap} ATAU {id: {'name':.., 'capacity':cap}} -> {res: cap}."""
    cap = {}
    for k, v in raw.items():
        cap[k] = v["capacity"] if isinstance(v, dict) else v
    return cap


def normalize_requirements(raw):
    """
    Terima {activity: {resource: qty}} (format resource_requirements.json)
    ATAU {'assignments': [{activity_id, resource_id, daily_demand}]} (format v3).
    Kembalikan {activity: {resource: qty}}.
    """
    if isinstance(raw, dict) and "assignments" in raw:
        req = {}
        for a in raw["assignments"]:
            req.setdefault(a["activity_id"], {})[a["resource_id"]] = a["daily_demand"]
        return req
    return raw


# --------------------------------------------------------------------------
# Problem RCTCT
# --------------------------------------------------------------------------
class RCTCTProblem(MultimodalProblem):
    problem_type = "Multimodal"

    def __init__(self, activity_data, resource_capacity, resource_requirements,
                 T_max, c_late=150.0, c_early=100.0,
                 budget=None, big_M=1e9, params=None):
        # --- simpan data ---
        self.activity_data = activity_data
        self.capacity = normalize_capacity(resource_capacity)
        self.requirements = normalize_requirements(resource_requirements)
        self.activities = list(activity_data.keys())

        self.preds = {a: list(activity_data[a].get("required_activities", []))
                      for a in self.activities}
        self.normal = {a: int(activity_data[a]["activity_normal_time"]) for a in self.activities}
        self.minimum = {a: int(activity_data[a]["activity_min_time"]) for a in self.activities}
        self.crash_cost = {a: float(activity_data[a]["crash_cost"]) for a in self.activities}

        # aktivitas yang bisa di-crash (d_min < d_max) -> hanya ini yg jadi variabel crash
        self.crashable = [a for a in self.activities if self.minimum[a] < self.normal[a]]

        # --- parameter objektif ---
        self.T_max = T_max
        self.c_late = c_late
        self.c_early = c_early
        self.budget = budget
        self.big_M = big_M

        # --- parameter SDOA (bisa dioverride) ---
        self._params = params or {
            "m_cluster": 500, "k_cluster": 10,
            "r_cl": 0.95, "theta_cl": np.pi / 4,
            "sdoa_m": 120, "sdoa_k_max": 150,
            "r": 0.95, "theta": np.pi / 4,
            "epsilon": 1e-5, "delta": 0.5,
            "gamma": -float("inf"), "num_check_points": 2,
        }

        # index blok crash & priority dalam vektor keputusan
        self.n_crash = len(self.crashable)
        self.n_prio = len(self.activities)
        super().__init__()  # memanggil get_info() -> set self.domain, self.n_var

    @property
    def name(self):
        return "RCTCT Baseline (Skenario 1, integer, serial-SGS)"

    # ---- domain box statis + params ----
    def get_info(self):
        domain = []
        # blok crash: (0, d_max - d_min) dgn margin 0.5 supaya rounding di tepi aman
        for a in self.crashable:
            hi = self.normal[a] - self.minimum[a]
            domain.append((-0.5, hi + 0.5))
        # blok priority: (0, 1) per aktivitas
        domain += [(0.0, 1.0)] * self.n_prio
        return domain, self._params

    # ---- pecah vektor keputusan jadi durasi + priority ----
    def _split(self, x):
        crash_raw = x[:self.n_crash]
        prio_raw = x[self.n_crash:]

        dur = {}
        for a in self.activities:
            dur[a] = self.normal[a]  # default: tak bisa di-crash -> durasi normal
        for j, a in enumerate(self.crashable):
            hi = self.normal[a] - self.minimum[a]
            c = int(round(crash_raw[j]))
            c = max(0, min(hi, c))  # clip ke [0, d_max-d_min]  -> durasi integer
            dur[a] = self.normal[a] - c

        priority = {a: float(prio_raw[j]) for j, a in enumerate(self.activities)}
        return dur, priority

    # ---- cek apakah menempatkan aktivitas di [t, t+d) melanggar kapasitas ----
    def _resource_ok(self, start, d, req, usage):
        for t in range(start, start + d):
            for k, need in req.items():
                if usage.get((t, k), 0) + need > self.capacity.get(k, 0):
                    return False
        return True

    # ======================================================================
    # DECODER: serial Schedule-Generation Scheme (tiga langkah)
    #   1. precedence -> earliest_start = max(finish predecessor)
    #   2. resource   -> geser start maju sampai kapasitas cukup
    #   3. durasi (sudah ditetapkan di _split dari vektor keputusan)
    # ======================================================================
    def decode(self, x):
        dur, priority = self._split(x)

        scheduled = {}                 # a -> (start, finish)
        usage = {}                     # (hari, resource) -> unit terpakai
        unscheduled = set(self.activities)

        while unscheduled:
            # aktivitas yang SEMUA predecessor-nya sudah terjadwal
            eligible = [a for a in unscheduled
                        if all(p in scheduled for p in self.preds[a])]
            if not eligible:
                # ada cycle di precedence -> tandai infeasible
                return None, float("inf"), dur
            # LANGKAH pemilihan urutan: priority key tertinggi menang saat rebutan
            a = max(eligible, key=lambda z: priority[z])

            # LANGKAH 1: precedence
            est = max((scheduled[p][1] for p in self.preds[a]), default=0)

            # LANGKAH 2: resource (geser maju sampai muat)
            req_a = self.requirements.get(a, {})
            t = est
            while not self._resource_ok(t, dur[a], req_a, usage):
                t += 1

            s_i, e_i = t, t + dur[a]
            scheduled[a] = (s_i, e_i)
            for tau in range(s_i, e_i):
                for k, need in req_a.items():
                    usage[(tau, k)] = usage.get((tau, k), 0) + need
            unscheduled.remove(a)

        makespan = max(e for _, e in scheduled.values())
        return scheduled, makespan, dur

    # ---- biaya total (yang ingin diminimalkan) ----
    def total_cost(self, x):
        scheduled, makespan, dur = self.decode(x)
        if scheduled is None:
            return self.big_M  # infeasible (cycle)

        crash_spend = sum(self.crash_cost[a] * (self.normal[a] - dur[a])
                          for a in self.activities)

        late = self.c_late * max(0, makespan - self.T_max)
        early = self.c_early * max(0, self.T_max - makespan)

        cost = crash_spend + late - early

        # budget constraint (soft, via penalty)
        if self.budget is not None and crash_spend > self.budget:
            cost += self.big_M * (crash_spend - self.budget)

        return cost

    # ---- fitness (pysne memaksimalkan) : -cost ----
    def g_func(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return -self.total_cost(x)
        # batch: engine kadang mengirim (m, n) -> kembalikan shape (m,)
        return np.array([-self.total_cost(row) for row in x])

    def evaluate_fitness(self, x):
        return self.g_func(x)

    # ----------------------------------------------------------------------
    # Override seleksi akhir.
    # Uji "puncak" bawaan MultimodalProblem menggeser tiap dimensi sebesar
    # epsilon; karena durasi kita INTEGER (dibulatkan), fitness datar terhadap
    # perturbasi kecil sehingga SEMUA titik lolos sebagai puncak -> salah.
    # Di sini cukup: buang yg di luar domain, dedupe by delta, ambil top-K.
    # ----------------------------------------------------------------------
    def select_final_optimal(self, candidates):
        domain, params = self.get_info()
        delta = params.get("delta", 0.5)
        scored = [(c, self.evaluate_fitness(c))
                  for c in candidates if is_in_domain(c, domain)]
        return filter_unique_roots(scored, delta)

    # alias supaya kompatibel dgn solver.solve_system yg memanggil select_final_roots
    def select_final_roots(self, candidates):
        return self.select_final_optimal(candidates)

    # ---- util: laporan solusi terbaik (biaya minimum) ----
    def report(self, x):
        scheduled, makespan, dur = self.decode(x)
        crash_plan = {a: self.normal[a] - dur[a]
                      for a in self.activities if dur[a] < self.normal[a]}
        return {
            "makespan": makespan,
            "total_cost": self.total_cost(x),
            "crash_days": crash_plan,          # aktivitas -> berapa hari dipercepat
            "schedule": scheduled,             # aktivitas -> (mulai, selesai)
        }


# --------------------------------------------------------------------------
# Contoh pemakaian (sesuaikan path ke data v3 milikmu)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from pysne.solver import solve_system

    activity_data = load_json("data/activity_data_v3.json")
    resource_capacity = load_json("data/resource_capacity_v3.json")
    resource_requirements = load_json("data/resource_requirements_v3.json")

    prob = RCTCTProblem(
        activity_data=activity_data,
        resource_capacity=resource_capacity,
        resource_requirements=resource_requirements,
        T_max=243,          # target tenggat (contoh Skenario 1)
        c_late=150.0,
        c_early=100.0,
        budget=None,        # isi angka untuk mengaktifkan budget constraint
    )

    hasil = solve_system(prob, prob.get_info()[1], verbose=True)
    optima = hasil["roots"]  # kumpulan solusi lokal (multimodal)

    if len(optima) == 0:
        print("Tidak ada solusi ditemukan; naikkan sdoa_m / m_cluster / k_max.")
    else:
        # pilih yg biaya paling rendah dari semua optimum lokal
        best = min(optima, key=prob.total_cost)
        rep = prob.report(best)
        print("Makespan  :", rep["makespan"])
        print("Total cost:", rep["total_cost"])
        print("Crash plan:", rep["crash_days"])