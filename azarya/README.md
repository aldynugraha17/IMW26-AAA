# SOAC untuk Project Crashing (Integer)

Adaptasi **Spiral Optimization Algorithm with Clustering (SOAC)** untuk masalah
*project crashing*: menentukan aktivitas mana yang dipercepat (di-*crash*) dan
berapa hari, agar **total biaya crashing minimum** sambil memenuhi **deadline
proyek** — dan menemukan **semua** kombinasi optimum dalam satu kali run.

Berbasis dua paper:

1. Sidarto, Kania, Sumarti (2017) — *Finding Multiple Solutions of Multimodal
   Optimization using Spiral Optimization Algorithm with Clustering*, MENDEL 23(1).
2. Sumarti, Sidarto, Kania, Edriani, Aditya (2023) — *A method for finding
   numerical solutions to Diophantine equations using SOAC*, Applied Soft
   Computing 145 (prosedur solusi **integer** via pelebaran domain ±0.5 +
   pembulatan).

Implementasi memakai pipeline [`pysne`](https://github.com/p2ms-optimization/pysne/tree/eksperimen)
(branch `eksperimen`) **tanpa mengubah satu baris pun** di pysne — cukup satu
subclass `BaseProblem`.

---

## 1. Formulasi Masalah

| Simbol | Arti |
|---|---|
| `n` | banyaknya task |
| `d_j` | **variabel keputusan**: durasi integer task `j`, `d_min_j ≤ d_j ≤ d_max_j` |
| `d_max_j` | durasi normal task `j` |
| `d_min_j` | durasi minimum setelah crashing |
| `y_j = d_max_j − d_j` | jumlah hari crash task `j` |
| `c_j` | biaya crashing per hari task `j` |
| `s_j`, `e_j` | waktu mulai / selesai (turunan, bukan variabel keputusan) |
| `T` | deadline proyek |

```
min   Z(d) = Σ_j c_j · y_j = Σ_j c_j · (d_max_j − d_j)
s.t.  s_j = max_{p ∈ pred(j)} e_p        (0 jika tanpa predecessor)
      e_j = s_j + d_j                    (forward pass / CPM, urutan topologis)
      makespan(d) = max_j e_j ≤ T
      d_j ∈ {d_min_j, ..., d_max_j}
```

Catatan penting: **tanpa kendala deadline, jawabannya trivial** (`d_j = d_max_j`,
biaya 0). Kendala `makespan ≤ T` inilah yang membuat masalah bermakna, dan
ditangani lewat penalti di fitness (bukan lewat kendala eksplisit).

### Transformasi ke fitness F (SOAC memaksimumkan F ∈ (0, 1])

```
Z_norm = Z(d) / Z_max,   Z_max = Σ_j c_j (d_max_j − d_min_j)
v      = max(0, makespan(d) − T)          # pelanggaran deadline

F(d) = 1 / (1 + Z_norm)          jika v = 0   → F ∈ [1/2, 1]   (feasible)
F(d) = 1 / (2 + Z_norm + v)      jika v > 0   → F <  1/2       (infeasible)
```

Dua sifat kunci: (a) titik feasible **selalu** mengalahkan titik infeasible mana
pun; (b) di wilayah infeasible tetap ada gradien menuju feasibility (v mengecil
→ F membesar). Semua vektor durasi berbiaya minimum sama membentuk **plateau F
maksimum** — struktur "banyak akar" yang persis dieksploitasi clustering SOAC,
sama seperti banyak akar pada persamaan Diophantine.

### Representasi integer (prosedur ala paper Diophantine)

Titik spiral `x` bergerak di ruang **kontinu**; dekode per dimensi `j`:

1. `unit_cube=True` (default): `x_j ∈ [0,1]` dipetakan affine ke
   `[d_min_j − 0.5, d_max_j + 0.5]`.
2. Bulatkan ke integer terdekat (`np.rint`).
3. **Clamp** ke `[d_min_j, d_max_j]` — wajib, karena titik tepat di tepi domain
   yang dilebarkan bisa terbulatkan keluar rentang (dan `np.round(6.5) = 6`,
   *banker's rounding*).

`unit_cube=False` = spiral langsung di `[d_min−0.5, d_max+0.5]^n` (pola
`DiophantineProblem` pysne). Keduanya ekuivalen secara sampling Sobol
(`generate_sobol_points` memang men-generate di `[0,1]^n` lalu men-skala);
bedanya hanya geometri rotasi spiral.

---

## 2. Alur Algoritma

```
┌─ FASE 1: CLUSTERING (pysne: perform_iterative_clustering) ──────────────┐
│ 1. Generate m_cluster titik Sobol x_i(0) di domain; k = 0.              │
│ 2. Evaluasi F(x_i): dekode → forward pass s,e → Z, makespan → F.        │
│ 3. Cluster pertama: pusat = argmax F, radius = ½·min(b_l − a_l).        │
│ 4. Tiap titik dengan F > γ·F_best yang bukan pusat cluster masuk        │
│    Function Cluster (kasus valley / mid-better / update-center).        │
│ 5. Rotasi spiral: x_i(k+1) = S_n(r,θ)·x_i(k) − (S_n − I)·x_best.        │
│ 6. Ulangi 4–5 sebanyak k_cluster kali.                                  │
└──────────────────────────────────────────────────────────────────────────┘
┌─ FASE 2: OPTIMISASI (pysne: run_sdoa_on_clusters) ──────────────────────┐
│ Per cluster: generate sdoa_m titik Sobol di [pusat ± radius] ∩ domain,  │
│ jalankan SDOA sdoa_k_max iterasi (fitness sama, tanpa clustering)       │
│ → satu kandidat per cluster.                                            │
└──────────────────────────────────────────────────────────────────────────┘
┌─ FASE 3: SELEKSI (custom: select_final_roots) ──────────────────────────┐
│ a. Dekode & buang kandidat infeasible (makespan > T).                   │
│ b. [local_search]  _polish: local search integer per kandidat.          │
│ c. Dedup per tuple integer; simpan yang Z ≤ Z_best + cost_tolerance.    │
│ d. [expand_plateau] _expand_plateau: BFS neutral-move untuk memanen     │
│    optimum alternatif yang mungkin terlewat.                            │
└──────────────────────────────────────────────────────────────────────────┘
```

Filter `1 − F < ε` ala Diophantine **tidak dipakai** — filter itu hanya benar
saat target F = 1 eksak (residual nol). Di sini optimum biayanya bukan nol,
jadi seleksinya berbasis "biaya sama dengan biaya terbaik yang ditemukan".

---

## 3. Mengapa Ada `_polish` dan `_expand_plateau`?

### Masalah yang dihadapi di dimensi tinggi

Pada instance kecil (5 task, rantai serial) SOAC murni sempurna: keempat
optimum ditemukan, tervalidasi brute force. Tapi pada data penuh **25 task**
(`activity_data_v3.json`, deadline 243), SOAC murni mentok di **Z = 200**,
padahal optimum eksak (via ILP) adalah **Z = 180**. Akar masalahnya:

1. **Kutukan dimensi pada sampling.** 2048 titik Sobol di 5 dimensi itu rapat;
   di 25 dimensi sangat renggang. Peluang ada titik awal dekat kombinasi
   optimum eksak kecil sekali.
2. **Landscape fitness berbentuk plateau bertingkat.** Karena evaluasi lewat
   pembulatan, F konstan di dalam tiap "kotak" integer. Spiral hanya melihat
   nilai F, dan banyak kotak berbeda memberi F hampir sama (selisih Z antar
   tetangga bisa cuma puluhan dari Z_max ribuan) — arah perbaikan jadi kabur.
3. **Optimum butuh koordinasi antar-koordinat.** Solusi Z=200 yang ditemukan
   SOAC meng-crash task mahal (Plumbing Trim, Cleaning); untuk turun ke Z=180
   harus *sekaligus* meng-uncrash task mahal **dan** meng-crash task murah
   (Site Work, Insulation, Final Punch-out) tanpa melanggar deadline. Gerakan
   "tukar" seperti ini hampir mustahil dihasilkan rotasi spiral + pembulatan,
   karena setiap langkah tunggalnya melewati titik dengan F lebih buruk atau
   infeasible.

### `_polish` (memetic local search)

Menambal kelemahan #3 dengan dua langkah integer eksak yang diulang sampai
stabil, diterapkan ke tiap kandidat hasil SDOA:

- **uncrash**: `d_j += 1` jika masih feasible → biaya *pasti* turun `c_j`
  (crashing yang tidak perlu dibuang);
- **swap**: `d_j += 1, d_k −= 1` jika feasible dan `c_j > c_k` → biaya turun
  `c_j − c_k` (pindahkan hari crash dari task mahal ke task murah).

SOAC tetap berperan sebagai **penjelajah global** (menemukan basin/wilayah
feasible yang beragam), polish sebagai **penajam lokal**. Hasil: Z = 180,
gap 0 terhadap ILP. Pola hybrid ini standar di literatur metaheuristik
(algoritma memetic). Matikan dengan `local_search=False` untuk membandingkan
SOAC "murni" ala paper.

### `_expand_plateau`

Tujuan utamanya *multiple solutions*. Polish bersifat menyatukan — kandidat
berbeda bisa konvergen ke optimum yang sama, sehingga optimum alternatif
tercecer. `_expand_plateau` melakukan BFS **neutral move**: swap satu hari
crash antara dua task dengan `c_j = c_k` (biaya tak berubah) yang tetap
feasible. Dari satu optimum, semua "saudara sebiayanya" yang terhubung lewat
langkah netral ikut terpanen (dibatasi `max_solutions`). Pada instance adiel
T=243 optimumnya terbukti **unik** (kapasitas crash seluruh task c=30 persis
6 hari: 2+1+3); pada instance dengan beberapa task berbiaya sama dan kapasitas
berlebih, fitur ini yang memastikan semua optimum terlaporkan.

---

## 4. Kamus Parameter

### Konstruktor `ProjectCrashingProblem(tasks, deadline, ...)`

| Parameter | Default | Arti |
|---|---|---|
| `tasks` | — | list of dict: `name`, `predecessors`, `d_min`, `d_max`, `crash_cost` |
| `deadline` | — | `T`, batas makespan (hari) |
| `params` | `{}` | override hyperparameter SOAC (tabel di bawah) |
| `unit_cube` | `True` | `True`: spiral di `[0,1]^n`, dekode di fitness; `False`: spiral di `[d_min−0.5, d_max+0.5]^n` |
| `cost_tolerance` | `0.0` | laporkan kandidat dengan `Z ≤ Z_best + toleransi` (0 = hanya minimum eksak) |
| `local_search` | `True` | aktifkan `_polish` di fase seleksi |
| `expand_plateau` | `True` | aktifkan `_expand_plateau` di fase seleksi |

### Hyperparameter SOAC (dict `params`)

| Kunci | Default | Fase | Arti & saran tuning |
|---|---|---|---|
| `m_cluster` | 256 | Clustering | jumlah titik Sobol awal. Naikkan drastis untuk dimensi tinggi (2048 untuk 25 task); idealnya pangkat 2 (sifat balance Sobol) |
| `k_cluster` | 12 | Clustering | banyaknya iterasi pembaruan cluster (rotasi + Function Cluster) |
| `gamma` | 0.85 | Clustering | cutoff relatif: hanya titik dengan `F > γ·F_best` yang boleh jadi pusat cluster. Kecilkan agar lebih banyak wilayah tereksplorasi (lebih banyak cluster, lebih lambat) |
| `r_cl` | 0.95 | Clustering | laju konvergensi spiral fase clustering, `0 < r < 1`; makin dekat 1 makin lambat mengerut (eksplorasi lebih lama) |
| `theta_cl` | π/4 | Clustering | sudut rotasi spiral fase clustering, `0 < θ < 2π` |
| `num_check_points` | 1 | Clustering | banyak titik uji di segmen y–x_C pada Function Cluster (1 = titik tengah, sesuai paper) |
| `sdoa_m` | 64 | Optimisasi | titik Sobol per cluster untuk SDOA |
| `sdoa_k_max` | 120 | Optimisasi | iterasi maksimum SDOA per cluster |
| `sdoa_r` | 0.95 | Optimisasi | laju konvergensi SDOA; naikkan (mis. 0.97) untuk intensifikasi lebih halus |
| `sdoa_theta` | π/4 | Optimisasi | sudut rotasi SDOA |
| `delta` | 0.4 | Seleksi | jarak minimum antar solusi di `filter_unique_roots` pysne. Di problem ini dedupe utama berbasis tuple integer, jadi `delta` jarang berpengaruh |
| `epsilon` | 1e-7 | Seleksi | toleransi residual; **tidak dipakai** untuk filter akhir di problem ini (lihat §2), hanya diteruskan ke engine |

Trade-off umum: `m_cluster`, `k_cluster`, `sdoa_m`, `sdoa_k_max` ↑ = kualitas
& kelengkapan solusi ↑, waktu komputasi ↑ (linear terhadap masing-masing).

---

## 5. Struktur File & Cara Pakai

```
project_crashing_problem.py   # ProjectCrashingProblem (subclass BaseProblem pysne)
run_crashing_soac.py          # contoh 5 task + validasi brute force
run_crashing_adiel.py         # data 25 task (activity_data_v3.json) + validasi ILP
visualize_crashing.py         # Gantt chart baseline vs SOAC (gaya adiel)
activity_data_v3.json         # data 25 task (format dict per nama aktivitas)
```

```bash
git clone -b eksperimen https://github.com/p2ms-optimization/pysne.git
export PYTHONPATH=/path/ke/pysne

python run_crashing_soac.py    # instance kecil, ground truth brute force
python run_crashing_adiel.py   # instance 25 task, ground truth ILP (scipy.milp)
python visualize_crashing.py   # menghasilkan gantt_soac_adiel.png
```

Penggunaan minimal:

```python
from pysne.solver import solve_system
from project_crashing_problem import ProjectCrashingProblem

problem = ProjectCrashingProblem(tasks, deadline=243,
                                 params={"m_cluster": 2048, "k_cluster": 15})
result = solve_system(problem, problem.get_info()[1], verbose=True)
print(problem.report(result["roots"]))
```

Format `activity_data_v3.json` → `tasks` (lihat `load_tasks` di
`run_crashing_adiel.py`): `required_activities` → `predecessors`,
`activity_min_time` → `d_min`, `activity_normal_time` → `d_max`,
`crash_cost` → `crash_cost` (`normal_cost` tidak dipakai objektif crashing).

---

## 6. Hasil

| Instance | Ground truth | SOAC murni | SOAC + polish | Multiple optima |
|---|---|---|---|---|
| 5 task serial, T=42 | brute force: Z*=220, 4 optimum | Z=220, 4/4 ditemukan | Z=220, 4/4 | ya (4) |
| 25 task adiel, T=243 | ILP: Z*=180 | Z=200 (gap 20) | **Z=180 (gap 0)** | unik (terverifikasi neutral-move) |

Optimum instance adiel: crash 6 hari seluruhnya pada task termurah di jalur
kritis (c=30/hari): Site Work −2, Insulation −1, Final Punch-out −3.

---

## 7. Arah Pengembangan

- Kendala kapasitas sumber daya harian (`resource_capacity_v3.json`,
  `resource_requirements_v3.json`) sebagai suku penalti kedua di fitness.
- Mode dinamis (T0, task selesai/berjalan dikunci) seperti model baseline IDSC.
- Sweep deadline T → kurva time–cost trade-off.
