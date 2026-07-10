# SPOC untuk Project Crashing dengan Kendala Kapasitas (RCPSP-TCT) — Variant B

Adaptasi **Spiral Optimization Algorithm with Clustering (SOAC/SPOC)** untuk
*integer project crashing* dengan kendala kapasitas sumber daya —
Resource-Constrained Project Scheduling Problem, Time-Cost Tradeoff
(**RCPSP-TCT**). Dibangun di atas pipeline `pysne` dan mengacu pada
Sidarto-Kania-Sumarti (2017) serta SOAC-Diophantine (2023).

Versi terkini adalah **Variant B (tanpa clipping)**: decode tidak memotong
titik spiral ke dalam domain, melainkan membiarkan spiral bereksplorasi bebas
dan menetralkan titik luar domain lewat fitness F = 0. Pada data uji 25 task,
varian ini adalah **satu-satunya yang berhasil menemukan ketiga solusi optimal
di T = 241** (termasuk sel tengah plateau yang tak pernah tertangkap varian
clipping), konsisten dengan temuan bahwa clipping "melipat" titik luar domain
menjadi artefak tepi ber-fitness tinggi yang meruntuhkan eksplorasi spiral.

---

## Isi folder

| File | Peran |
|------|-------|
| `project_crashing_problem_newest_compare.py` | Kelas problem **terbaru** (Variant B + RCPSP + seluruh optimasi kecepatan) |
| `run_crashing_adiel_new.py` | ILP ground truth (RCPSP-TCT) + runner SOAC untuk satu deadline |
| `run_sweep_tuner_rcpsp.py` | Sweep deadline + grid search parameter (cache jadwal dibagikan lintas T) |
| `activity_data_v3.json` | 25 task konstruksi (precedence, durasi normal/min, crash cost) |
| `resource_capacity_v3.json` | Kapasitas 24 jenis sumber daya |
| `resource_requirements_v3.json` | Kebutuhan sumber daya tiap task |

---

## Formulasi masalah

Variabel keputusan adalah durasi integer `d = (d_1, …, d_n)` dengan
`d_min_j ≤ d_j ≤ d_max_j` (d_max = durasi normal, d_min = durasi tersingkat
setelah crashing). Objektifnya meminimumkan total biaya crashing
`Z(d) = Σ_j c_j · (d_max_j − d_j)`, terhadap empat kendala: (1) precedence,
(2) integralitas, (3) deadline `makespan(d) ≤ T`, dan (4) **kapasitas sumber
daya** — untuk tiap resource r dan tiap satuan waktu, total permintaan task
yang aktif tidak boleh melebihi kapasitas C_r.

### Penanganan kendala (tiga mekanisme, tanpa mengubah algoritma spiral)

Paper rujukan menyelesaikan masalah tanpa kendala; di sini masalah berkendala
ditransformasi menjadi maksimisasi tanpa kendala yang himpunan optimumnya
identik, lalu SPOC dijalankan murni tanpa modifikasi:

1. **Kendala kotak & integer — decoder.** Titik spiral x hidup di ruang
   kontinu; decode memetakan affine ke `[d_min−0.5, d_max+0.5]` lalu
   membulatkan (prosedur integer SOAC-Diophantine). Kendala dipenuhi oleh
   konstruksi representasi.
2. **Precedence & kapasitas — konstruksi jadwal.** Jadwal dibangun oleh
   CPM/SGS yang menghormati precedence dan kapasitas; jadwal ilegal tidak
   pernah eksis di pipeline. Kapasitas tidak menambah suku penalti apa pun —
   konsekuensinya (jadwal memanjang) mengalir alami ke makespan.
3. **Deadline — penalti berlapis pada fitness** (lihat di bawah). Secara
   konseptual setara feasibility rules Deb (2000) yang disandikan ke satu
   skalar; taksonomi umum di Coello Coello (2002).

---

## Fungsi fitness (tiga pita)

```
Z_norm = Z / Z_max,   Z_max = Σ_j c_j (d_max_j − d_min_j)
v      = max(0, makespan_resource − T)

F(d) = 1 / (1 + Z_norm)        feasible          →  F ∈ [½, 1]
F(d) = 1 / (2 + Z_norm + v)    infeasible        →  F ≤ ⅓
F(d) = 0                       d luar [d_min,d_max] →  bukan kandidat
```

Sifat-sifat kunci:

- **Pemisahan total.** Feasible terburuk (½) selalu di atas infeasible
  terbaik (≤ ⅓); zona (⅓, ½) tak pernah dihuni. Titik luar domain (F = 0)
  tak pernah memenangkan argmax, sehingga pusat spiral `x_p` selalu tetap
  menunjuk ke dalam domain — inilah jantung Variant B.
- **Jangkar bermakna fisik.** F = 1 ⟺ tanpa crashing (Z = 0);
  F = ½ ⟺ crashing maksimal (Z = Z_max, karena Z_max didefinisikan sebagai
  biaya full-crash sehingga ternormalisasi tepat ke 1).
- **Gradien dua arah.** Di wilayah feasible F memeringkat biaya (argmax F ≡
  argmin Z, transformasi monoton); di wilayah infeasible suku v memberi
  kompas menuju feasibility.
- **Plateau = "banyak akar".** Semua solusi berbiaya minimum berbagi F
  maksimum yang sama — struktur multimodal yang isomorfik dengan banyak akar
  pada pipeline Diophantine, sehingga mesin clustering menemukan banyak
  solusi optimal sekaligus.
- **F maksimum tidak diketahui di muka** (= 1/(1+Z\*/Z_max), butuh Z\* yang
  justru sedang dicari). Konsekuensi: kriteria absolut tidak bisa dipakai —
  lihat catatan parameter.

---

## Pengecekan domain tiga lapis (Variant B)

Decode adalah transformasi **murni** x → d, tanpa gerbang dan tanpa clip.
Pengecekan domain hidup di tiga lapis:

1. **Fase clustering (built-in pysne).** `is_in_domain(x, domain)` menyaring
   titik luar unit cube dari pembenihan cluster. Di fase SDOA pysne *tidak*
   memanggil cek ini — pelindungnya lapis 2.
2. **`evaluate_fitness` (tervektorisasi).** Dua mask se-batch:
   `in_cube` di ruang-x terhadap `[0,1]^n`, dan `in_dom` di ruang-d terhadap
   `[d_min, d_max]`; F = 0 bagi yang gagal, dan titik gagal tak pernah masuk
   SGS. Cek ruang-d **mensubsumsi** cek ruang-x (x ketat di luar [0,1] pasti
   membulat keluar rentang) dan lebih kuat: titik tepi x = 0/1 bisa membulat
   keluar akibat banker's rounding `np.rint` (pada data ini, x = 0 — titik
   Sobol pertama — membulat keluar pada 15 dari 25 task) dan hanya tertangkap
   lapis ruang-d. Lapis `in_cube` dipertahankan eksplisit sebagai kesesuaian
   dengan pola pysne dan pengaman bila pemetaan decode kelak berubah.
   Biaya kedua mask ≈ 0,03% dari biaya SGS (≈ 100 ns/titik).
3. **Seleksi akhir.** Kandidat yang decode-nya keluar `[d_min, d_max]`
   dibuang sebelum cek feasibility dan biaya.

Tanpa pagar ini, un-clip mentah menimbulkan tiga lubang yang semuanya pernah
teramati: durasi negatif → crash `max() empty` di SGS; d > d_max → Z_norm
negatif → pembagian nol / F > 1 (attractor palsu di luar domain); dan
korupsi diam-diam (jadwal mustahil dinilai feasible).

---

## Penegakan kapasitas: makespan resource-feasible

Dengan sumber daya, makespan dihitung lewat **Schedule Generation Scheme**:
serial SGS berurutan prioritas **LFT** (paling genting dulu), dengan
**parallel SGS** sebagai pendapat kedua. Sifat kunci
`makespan_resource(d) ≥ makespan_CPM(d)` dipakai dua arah: sebagai
pre-filter (CPM > T ⟹ pasti infeasible, SGS dilewati) dan sebagai
short-circuit (serial mencapai batas bawah CPM ⟹ terbukti optimal, parallel
dilewati). SGS adalah heuristik batas-atas; pada instans ini terverifikasi
persis sama dengan RCPSP eksak CP-SAT (gap 0 pada 60 vektor durasi acak),
sehingga optimum sejati tidak pernah salah ditolak — klaim ini per-instans
dan diverifikasi ulang bila datanya berganti.

### Optimasi kecepatan (semua lossless; end-to-end ≈ 3× lebih cepat)

- **Analisis resource pengikat (eksak).** Task yang aktif bersamaan selalu
  membentuk antichain pada DAG precedence; resource yang permintaan
  antichain-maksimumnya ≤ kapasitas (ILP kecil sekali jalan saat init)
  mustahil dilanggar dan diabaikan total. Pada data ini hanya **3 dari 24**
  resource yang pengikat (G.C. Labor Crew, Tile Contractor, G.C. Finish
  Carpenter Crew).
- **Cache-first batch + dedup.** `evaluate_fitness` mengonsultasi memo dulu
  (kunci bytes per vektor durasi), mendeduplikasi baris baru (`np.unique`),
  dan hanya menghitung yang benar-benar baru — fase SDOA yang mengerucut
  praktis 100% cache-hit.
- **Cek pelanggaran tervektorisasi.** Titik baru dicek pelanggarannya pada
  jadwal CPM earliest-start sekaligus se-batch (difference-array + cumsum);
  yang bersih langsung selesai tanpa SGS.
- **SGS hybrid.** Penempatan dicoba langsung di earliest start (satu cek
  murah); pencarian slot penuh (sliding-window-max) hanya saat gagal.
- **Cache jadwal lintas deadline.** Jadwal SGS terbukti tidak bergantung
  deadline (pergeseran Tref menggeser semua LF seragam → urutan LFT tak
  berubah), sehingga satu cache (`sched_cache`) aman dibagikan lintas
  deadline dan konfigurasi pada sweep — deadline kedua dan seterusnya
  mewarisi seluruh memo.

Skala biaya per titik: CPM tervektorisasi ≈ 0,001 ms; SGS ≈ 0,26–0,75 ms —
inilah sumber kenaikan waktu dibanding model tanpa sumber daya, dan alasan
parameter populasi perlu ditimbang terhadap waktu.

---

## ILP ground truth

`ilp_ground_truth` bercabang otomatis (`scipy.optimize.milp` / HiGHS): tanpa
sumber daya memakai LP start-kontinu ringan; dengan sumber daya memakai MILP
**time-indexed multi-mode** — biner `u[j,t,δ] = 1` bila task j mulai di t
dengan durasi δ, kapasitas ditegakkan per (resource, waktu) — eksak, dengan
ruang variabel dipersempit jendela start `[ES, LS]` dari CPM (≈ 8.000 biner,
belasan detik).

---

## Seleksi akhir (empat saringan)

Input: satu kandidat `x*` per cluster hasil SDOA. Berurutan: (1) decode +
saring domain integer; (2) saring feasibility via makespan resource-feasible;
(3) dedup eksak berkunci `tuple(d)`; (4) saring optimalitas — semua survivor
dengan `Z ≤ best_cost + tol` disimpan. Toleransi `tol = 1e-9 × max(1, Z_best)`
adalah toleransi **numerik** untuk meredam noise floating-point, bukan
toleransi optimisasi: gap Z antar solusi integer minimal 40 (unit biaya
terkecil × 1 hari), sepuluh orde di atas tol, sehingga nilai persisnya tidak
sensitif. Saringan (4) inilah yang membuat banyak solusi optimal dilaporkan
sekaligus.

---

## Parameter: mana yang hidup, mana yang vestigial

| Parameter | Status | Catatan |
|---|---|---|
| `gamma` | **Hidup, relatif** | Ambang pembenihan cluster `cutoff = γ × F_best` (cabang non-SNE pysne). Self-calibrating terhadap ketinggian plateau — γ = 0,85 yang sama bekerja di semua deadline tanpa disetel. Menaikkan γ memangkas jumlah cluster (lebih cepat, cakupan plateau menipis) — tuas utama bila cluster meledak pada Variant B. |
| `m_cluster`, `k_cluster` | Hidup | Skala fase clustering; keragaman cluster menentukan berapa sel plateau tertangkap. |
| `sdoa_m`, `sdoa_k_max`, `sdoa_r`, `sdoa_theta` | Hidup | Skala/laju SDOA per cluster. SDOA selalu berjalan `sdoa_k_max` penuh (lihat epsilon). |
| `num_check_points` | Hidup | Empiris menolong konvergensi di deadline sulit. |
| `epsilon` | **Vestigial** | Hanya dibaca early-stop SDOA berpagar `equations is not None` — kelas ini menyetel `equations = None`. Secara prinsip pun tak bisa dipakai: kriteria `1 − F ≤ ε` butuh target F yang diketahui (SNE: F = 1); pada optimisasi target itu justru yang dicari. |
| `delta` | **Vestigial** | Dulu untuk merge kandidat di seleksi akhir pysne yang kini dinonaktifkan; seleksi digantikan `select_final_optimal` milik kelas ini. |

Sinergi yang perlu diketahui: begitu populasi memuat satu titik feasible
(F ≥ ½), cutoff γ·F_best > ⅓ ≥ F infeasible mana pun — titik infeasible tak
pernah membenihkan cluster. Pita fitness dan mekanisme gamma saling mengunci.

---

## Validasi (data 25 task)

- ILP ground truth RCPSP-TCT: **Z\* = 260** di T = 241; kapasitas terbukti
  *inert* pada instans/deadline ini (Z\* dan ketiga optimum identik dengan
  model CPM — menambah kendala tak pernah menciptakan solusi baru).
- **Variant B menemukan 3/3 solusi optimal di T = 241**
  (m_cluster = 2048, num_check_points = 2), termasuk sel tengah
  (Plumbing Trim, Cleaning) = (4,13) yang merupakan interior plateau dan tak
  pernah tertangkap varian clipping. Ketiga sel: (5,12), (4,13), (3,14),
  semuanya Z = 260, F = 0,9706.
- Scheduler SGS cocok persis dengan CP-SAT (gap makespan 0 / 60 vektor acak);
  jadwal yang dilaporkan resource-feasible (mis. Concrete Slabs ter-level
  dari s = 47 ke s = 77 menghindari bentrok G.C. Labor Crew, permintaan 4 >
  kapasitas 3).
- Optimasi kecepatan diverifikasi bit-identik terhadap implementasi rujukan
  pada ribuan titik acak; end-to-end `solve_system` 163,8 s → 55,4 s (3,0×)
  dengan hasil identik.

## Batasan yang jujur

1. **Stochastic, tanpa jaminan kelengkapan.** Sel interior plateau bukan
   attractor spiral; parameter menaikkan peluang, bukan menjamin. Temuan 3/3
   perlu direplikasi lintas seed. Jaminan kelengkapan hanya dari enumerasi
   plateau deterministik pasca-SOAC (sengaja tidak diimplementasikan demi
   kemurnian algoritmik) atau solver eksak.
2. **SGS batas atas heuristik** — eksak pada instans ini, tidak dijamin
   universal.
3. **Kapasitas inert di data ini** — nilai penambahannya adalah kebenaran
   metodologis, Gantt yang benar-benar feasible, dan komparabilitas dengan
   solver RCPSP-TCT eksak; deadline lebih ketat berpotensi membuatnya
   mengikat.
4. **Kompatibilitas mundur**: tanpa `resource_capacity`/`resource_requirements`
   kelas kembali ke CPM murni (`ProjectCrashing[CPM]` vs
   `ProjectCrashing[RCPSP]`).

---

## Cara menjalankan

```bash
pip install "git+https://github.com/p2ms-optimization/pysne.git@eksperimen-int"
python run_crashing_adiel_new.py        # satu deadline
python run_sweep_tuner_rcpsp.py         # sweep T + grid search
```

Ketiga file JSON harus sefolder dengan runner (path relatif via
`Path(__file__).resolve().parent`). Catatan: pencarian tidak di-seed; hasil
antar-run dapat berbeda — untuk replikasi temuan, jalankan beberapa kali atau
tetapkan `np.random.seed` di runner.

## Rujukan

- Sidarto, K. A., Kania, A. (2017). *Spiral Optimization Algorithm with Clustering.* JACIII.
- Sumarti, N., dkk. (2023). *SOAC untuk sistem persamaan Diophantine.* Applied Soft Computing.
- Deb, K. (2000). *An efficient constraint handling method for genetic algorithms.*
- Coello Coello, C. A. (2002). *Theoretical and numerical constraint-handling techniques.*
