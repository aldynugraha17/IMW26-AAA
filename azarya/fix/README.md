# SPOC untuk Project Crashing (Integer) — Versi Terbaru

Adaptasi **Spiral Optimization Algorithm with Clustering (SPOC)** untuk *project
crashing*: memilih task mana yang dipercepat (di-*crash*) dan berapa hari agar
**total biaya crashing minimum** sambil memenuhi **deadline proyek**, dengan
menemukan **semua** kombinasi durasi optimum sekaligus.

Berbasis Sidarto-Kania-Sumarti (2017, SPOC multimodal) + Sumarti dkk. (2023,
prosedur integer Diophantine: pelebaran domain ±0.5 + pembulatan). Berjalan di
atas pipeline [`pysne`](https://github.com/p2ms-optimization/pysne/tree/eksperimen)
(branch `eksperimen`) **tanpa mengubah pysne** — hanya satu subclass `BaseProblem`.

File utama: `project_crashing_problem_new.py` (kelas `ProjectCrashingProblem`).

---

## 1. Apa yang Baru di Versi Ini

Empat perubahan terhadap versi sebelumnya:

1. **Evaluasi fitness tervektorisasi (batch).** `evaluate_fitness` menerima
   populasi (B, n) sekaligus. Forward pass CPM dijalankan untuk seluruh populasi
   dengan loop Python hanya atas task (`_forward_pass_batch`), sisanya operasi
   numpy sepanjang sumbu populasi — jauh lebih cepat pada `m_cluster` besar.

2. **Memori pasangan real–integer.** `evaluate_fitness(x, remember=True)`
   menyimpan `{x_real, d_int, Z, F, feasible}` ke `self.memory`, sehingga
   pemetaan titik real → solusi integer bisa ditelusuri untuk verifikasi/laporan.

3. **Varian B: tanpa clamp, solusi invalid ditolak (bukan dipalsukan).**
   `decode` tidak lagi menjepit hasil ke `[d_min, d_max]`. Titik dibiarkan apa
   adanya sesuai posisi spiral; jika pembulatannya keluar rentang, ia (a) diberi
   `F = 0` di `evaluate_fitness` selama iterasi -- diabaikan sebagai kandidat
   pusat tanpa memindahkan titik realnya; dan (b) ditolak oleh `_in_domain` di
   seleksi akhir. Solusi invalid **ditolak secara eksplisit**, bukan disamarkan
   ke tepi domain (lebih jujur secara semantik untuk pelaporan). Lihat §4.

---

## 2. Formulasi

| Simbol | Arti |
|---|---|
| `d_j` | variabel keputusan: durasi integer task `j`, `d_min_j ≤ d_j ≤ d_max_j` |
| `d_max_j`, `d_min_j` | durasi normal / minimum |
| `y_j = d_max_j − d_j` | hari crash |
| `c_j` | biaya crash per hari |
| `s_j`, `e_j` | mulai / selesai (turunan forward pass) |
| `T` | deadline |

```
min   Z(d) = Σ_j c_j (d_max_j − d_j)
s.t.  s_j = max_{p∈pred(j)} e_p ;  e_j = s_j + d_j   (forward pass, urutan topologis)
      makespan(d) = max_j e_j ≤ T
      d_j ∈ {d_min_j, ..., d_max_j}
```

Tanpa kendala deadline jawabannya trivial (`d_j = d_max_j`, biaya 0); kendala
`makespan ≤ T` yang membuatnya bermakna, ditangani lewat penalti di fitness.

### Fitness F (SPOC memaksimumkan F ∈ (0,1])

```
Z_norm = Z / Z_max,   Z_max = Σ_j c_j (d_max_j − d_min_j)
v      = max(0, makespan − T)

F = 1 / (1 + Z_norm)          jika feasible (v = 0)    → F ∈ [1/2, 1]
F = 1 / (2 + Z_norm + v)      jika infeasible (v > 0)  → F <  1/2
F = 0                         jika titik keluar domain integer
```

Angka **2** pada penyebut infeasible menjamin F < ½; angka **1** pada feasible
menjamin F ≥ ½. Selisih inilah **jurang di F = ½** yang membuat solusi feasible
terburuk pun mengalahkan infeasible terbaik ("penuhi deadline dulu, lalu
murahkan"). Suku `+ v` memberi gradien di wilayah infeasible (telat 1 hari lebih
baik dari telat 10) agar spiral bisa merambat menuju feasibility.

### Alur evaluasi satu titik (urutan yang benar)

```
titik real x  →  skala [0,1] ke [d_min−0.5, d_max+0.5]  →  BULATKAN (rint)
              →  d integer  →  cek domain (F=0 bila keluar, Varian B)
              →  forward pass CPM (s, e, makespan)  →  Z, v  →  F
```

Pembulatan terjadi **sebelum** forward pass (forward pass harus bekerja pada
durasi integer, bukan pecahan). Titik real tidak pernah ditimpa versi
integernya — pembulatan hanya untuk *menilai*.

---

## 3. Alur Algoritma (tiga fase, pysne tak diubah)

```
FASE 1 — CLUSTERING (perform_iterative_clustering)
  1. m_cluster titik Sobol di domain; k = 0.
  2. Evaluasi F (skala→bulatkan→forward pass; F=0 bila keluar domain).
  3. Cluster pertama: pusat = argmax F, radius = ½·min(b_l − a_l).
  4. Titik dgn F > γ·F_best (bukan pusat) → Function Cluster (valley/mid/update).
  5. Rotasi spiral semua titik ke pusat terbaik.
  6. Ulangi 4–5 sebanyak k_cluster kali.

FASE 2 — OPTIMISASI (run_sdoa_on_clusters)
  Per cluster: sdoa_m titik Sobol di [pusat ± radius] ∩ domain,
  SDOA sdoa_k_max iterasi (fitness sama, tanpa clustering) → 1 kandidat/cluster.

FASE 3 — SELEKSI (select_final_roots)
  Dekode → buang yang keluar domain (_in_domain) → buang infeasible
  → dedup tuple integer → simpan Z ≤ Z_best + cost_tolerance.
```

Titik real dirotasi sepanjang iterasi; pembulatan hanya untuk evaluasi. Di
seleksi akhir barulah rounding dipakai untuk menghasilkan output integer final.

---

## 4. Penanganan Domain (Varian B) & `_in_domain` vs `is_in_domain`

**Varian B** memilih *menolak* alih-alih *menjepit* titik di luar domain, dengan
dua lapis pertahanan sederhana:

- **Selama iterasi (`evaluate_fitness`).** Titik yang pembulatannya keluar
  `[d_min, d_max]` diberi `F = 0`, sehingga tidak akan pernah terpilih sebagai
  pusat spiral/cluster. Titik realnya tidak dipindahkan -- geometri spiral tetap
  murni.
- **Di seleksi akhir (`select_final_roots`).** Tiap kandidat final dicek
  `_in_domain`; yang keluar rentang dibuang sebelum dedup. Biaya O(jumlah_solusi
  × n_task) -- praktis gratis (mis. 5 solusi × 25 task = 125 perbandingan,
  sekali di ujung).

Kenapa "tolak" bukan "clamp"? Clamp *memalsukan* titik: durasi 11 (di luar
d_max=10) diam-diam dilaporkan sebagai 10, seolah itu yang ditemukan algoritma.
Varian B tidak menyamarkan -- titik invalid ditolak apa adanya, lebih bersih
dipertanggungjawabkan di laporan.

### `_in_domain` (kelas ini) vs `is_in_domain` (pysne) -- komplementer

| | `is_in_domain` (pysne.utils) | `_in_domain` (kelas ini) |
|---|---|---|
| Ruang | titik **real** | vektor **integer** hasil pembulatan |
| Batas | domain spiral: `[0,1]` atau `[d_min−0.5, d_max+0.5]` | rentang durasi `[d_min, d_max]` |
| Waktu | **sebelum** pembulatan | **sesudah** pembulatan |
| Peran | menjaga lintasan spiral di kotak pencarian | menyaring durasi hasil terjemahan agar sah |
| Dipakai di | clustering, engine SDOA, base pysne (hulu) | evaluate_fitness (F=0) & select_final_roots (hilir) |

Keduanya **berbeda ruang** dan **komplementer**, bukan duplikat: `is_in_domain`
menjaga *lintasan* real di hulu; `_in_domain` menyaring *hasil* integer di hilir.
Karena keduanya di ruang berbeda, `_in_domain` tidak bisa digantikan
`is_in_domain` -- cek pasca-pembulatan wajib dilakukan di ruang integer.

**Catatan cakupan praktis.** Untuk `unit_cube=True`, pysne sudah menyaring titik
di luar `[0,1]^n` di hulu, sehingga solusi keluar domain jarang muncul dan
`_in_domain` sebagian besar berfungsi sebagai *jaring pengaman* eksplisit
(assertion murah yang mendokumentasikan bahwa domain memang dipedulikan di
ujung). Ia benar-benar menolak kandidat hanya bila titik lolos hulu namun
membulat keluar rentang -- lebih relevan saat bereksperimen dengan
`unit_cube=False` atau margin non-standar.

## 5. Kamus Parameter

### Konstruktor
| Parameter | Default | Arti |
|---|---|---|
| `tasks` | — | list dict: `name`, `predecessors`, `d_min`, `d_max`, `crash_cost` |
| `deadline` | — | `T`, batas makespan |
| `params` | `{}` | override hyperparameter SPOC (tabel bawah) |
| `unit_cube` | `True` | `True`: spiral di `[0,1]^n`, dekode di fitness; `False`: spiral di `[d_min−0.5, d_max+0.5]^n` |
| `cost_tolerance` | `0.0` | laporkan kandidat `Z ≤ Z_best + toleransi` (0 = minimum eksak) |

### Hyperparameter SPOC (dict `params`)
| Kunci | Default | Fase | Catatan |
|---|---|---|---|
| `m_cluster` | 32768 | Clustering | titik Sobol awal. Besar untuk dimensi tinggi; idealnya pangkat 2 |
| `k_cluster` | 12 | Clustering | iterasi pembaruan cluster |
| `gamma` | 0.85 | Clustering | cutoff relatif `F > γ·F_best`; kecilkan → lebih banyak cluster |
| `r_cl` | 0.95 | Clustering | laju konvergensi spiral (0<r<1) |
| `theta_cl` | π/4 | Clustering | sudut rotasi |
| `num_check_points` | 1 | Clustering | titik uji Function Cluster (1 = titik tengah) |
| `sdoa_m` | 1024 | Optimisasi | titik Sobol per cluster |
| `sdoa_k_max` | 300 | Optimisasi | iterasi maksimum SDOA per cluster |
| `sdoa_r` | 0.97 | Optimisasi | laju konvergensi SDOA |
| `sdoa_theta` | π/4 | Optimisasi | sudut rotasi SDOA |
| `delta` | 0.4 | Seleksi | jarak dedup pysne; di sini dedup utama berbasis tuple integer, jadi jarang berpengaruh |
| `epsilon` | 1e-9 | Seleksi | inert di problem ini (`equations=None` mematikan early-stopping residual) |

Default `m_cluster=32768`, `sdoa_m=1024` disetel untuk instance 25 task agar
mencapai optimum eksak (~16 menit); turunkan untuk instance kecil / iterasi cepat.

---

## 6. Tantangan Dimensi Tinggi: Plateau Bertingkat

Karena F dievaluasi lewat pembulatan, landscape berbentuk **plateau bertingkat**:
F konstan di dalam tiap "kotak" integer, berubah hanya saat menyeberang batas
kotak. Di dalam plateau tidak ada gradien, dan beda F antar plateau tetangga
bisa sangat tipis — sehingga di dimensi tinggi SPOC dengan anggaran kecil bisa
berhenti di plateau suboptimal (mis. T=243, `m_cluster=2048` → Z=200 vs optimum
180).

Solusi murni-SPOC: **naikkan anggaran sampling**. `m_cluster=32768`,
`sdoa_m=1024` menutup gap ke 0 (mis. T=241 → Z=260 = optimum ILP). Kepadatan
Sobol lebih tinggi menaikkan peluang ada titik awal di basin optimum global.
Efek "penjepretan ke kisi" akibat pembulatan justru membuat target berupa titik
kisi berhingga (bukan vertex kontinu berukuran nol), sehingga sampling padat
efektif — lihat README varian kontinu untuk kontras lengkapnya.

Catatan struktural: optimum integer yang bertetangga dengan fitness sama dan
tanpa lembah di antaranya (mis. dua optimum di T=242 yang beda hanya satu swap
antar task berbiaya sama) sulit dipisahkan Function Cluster, sehingga tidak
semua anggota himpunan optimum selalu tertangkap. Diversifikasi geometri
(jalankan `unit_cube=True` dan `False`, atau ubah `theta`/`r`, lalu gabungkan
hasil) membantu tanpa mengubah algoritma.

---

## 7. Struktur File & Cara Pakai

```
project_crashing_problem_new.py  # ProjectCrashingProblem (versi terbaru)
run_crashing_adiel.py            # data 25 task + validasi ILP (scipy.milp)
visualize_crashing_new.py        # Gantt baseline vs optimized (gaya adiel)
activity_data_v3.json            # data 25 task
```

```bash
git clone -b eksperimen https://github.com/p2ms-optimization/pysne.git
export PYTHONPATH=/path/ke/pysne
python run_crashing_adiel.py
```

```python
from pysne.solver import solve_system
from project_crashing_problem_new import ProjectCrashingProblem

problem = ProjectCrashingProblem(tasks, deadline=241)
result = solve_system(problem, problem.get_info()[1], verbose=True)
print(problem.report(result["roots"]))
```

Format `activity_data_v3.json` → `tasks`: `required_activities`→`predecessors`,
`activity_min_time`→`d_min`, `activity_normal_time`→`d_max`,
`crash_cost`→`crash_cost` (`normal_cost` tak dipakai objektif crashing).

### Memakai memori pasangan real–integer
```python
problem.memory.clear()
F = problem.evaluate_fitness(X_batch, remember=True)   # X_batch: (B, n) real
for m in problem.memory:
    print(m["x_real"], "→", m["d_int"], "Z =", m["Z"], "F =", m["F"], m["feasible"])
```
Catatan: solver pysne memanggil `evaluate_fitness` tanpa `remember`, jadi memori
hanya terisi saat Anda memanggilnya sendiri dengan `remember=True` (untuk
verifikasi/analisis), bukan otomatis selama run.

---

## 8. Ground Truth & Hasil

Validasi memakai **ILP** (`scipy.optimize.milp`, eksak) karena brute force 25
task mustahil. Model linear (objektif + precedence + deadline linear, `d_j`
integer, `s_j` kontinu) → ILP memberi optimum eksak sebagai acuan.

| Instance | ILP Z* | SPOC (anggaran besar) |
|---|---|---|
| 5 task serial, T=42 | 220 (4 optimum, brute force) | Z=220, 4/4 |
| 25 task, T=241 | 260 | Z=260 (gap 0) |
| 25 task, T=242 | 220 (2 optimum) | Z=220 (gap 0), 1 dari 2 tertangkap |

ILP memberi **satu** optimum; SPOC mengejar **semua** — nilai tambah pendekatan
multimodal. Untuk multiplisitas yang tak lengkap (mis. T=242), lihat catatan
diversifikasi di §6.

---

## 9. Arah Pengembangan

- Kendala kapasitas sumber daya harian sebagai penalti kedua di fitness (di sini
  ILP linear tak lagi memadai — nilai tambah SPOC menguat).
- Sweep deadline T → kurva time–cost trade-off.
- Hook opsional agar `remember=True` aktif otomatis selama run untuk audit penuh.