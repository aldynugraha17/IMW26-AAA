# SOAC untuk Project Crashing dengan Kendala Kapasitas (RCPSP-TCT)

Adaptasi **Spiral Optimization Algorithm with Clustering (SOAC)** untuk menyelesaikan
*integer project crashing* **dengan kendala kapasitas sumber daya** — yaitu
Resource-Constrained Project Scheduling Problem, Time-Cost Tradeoff (**RCPSP-TCT**).
Dibangun di atas pipeline `pysne` (branch `eksperimen-int`) dan mengacu pada
Sidarto-Kania-Sumarti (2017) serta SOAC-Diophantine (Sumarti dkk., 2023).

Versi ini adalah pengembangan dari model *crashing* murni (CPM, tanpa sumber daya):
durasi tetap menjadi variabel keputusan, tetapi makespan kini dihitung lewat jadwal
yang **menghormati kapasitas sumber daya**, bukan sekadar jalur kritis.

---

## Isi folder

| File | Peran |
|------|-------|
| `project_crashing_problem_new.py` | Definisi `ProjectCrashingProblem` (subclass `BaseProblem` pysne) |
| `run_crashing_adiel_new.py` | ILP ground truth (RCPSP-TCT) + runner SOAC |
| `activity_data_v3.json` | 25 task konstruksi (precedence, durasi normal/min, crash cost) |
| `resource_capacity_v3.json` | Kapasitas tiap jenis sumber daya (24 jenis) |
| `resource_requirements_v3.json` | Kebutuhan sumber daya tiap task |

---

## Formulasi masalah

**Variabel keputusan.** Durasi integer `d = (d_1, ..., d_n)`, dengan
`d_min_j ≤ d_j ≤ d_max_j`. `d_max_j` = durasi normal, `d_min_j` = durasi tersingkat
setelah crashing.

**Objektif.** Minimkan total biaya crashing

```
Z(d) = Σ_j c_j · (d_max_j − d_j)
```

**Kendala.**
1. Precedence: task tidak boleh mulai sebelum seluruh pendahulunya selesai.
2. Integer: `d_j ∈ ℤ`.
3. Deadline: `makespan(d) ≤ T_deadline`.
4. **Kapasitas sumber daya (baru):** untuk tiap sumber daya `r` dan tiap satuan waktu,
   total permintaan task yang sedang aktif tidak boleh melebihi kapasitas `C_r`.

Kendala (4) inilah yang mengubah masalah dari CPM murni menjadi RCPSP.

---

## Bagaimana kapasitas ditegakkan

### Pada makespan (untuk SOAC)

Tanpa sumber daya, `makespan = forward-pass CPM` (earliest start, hanya precedence),
dan bisa divektorkan penuh atas seluruh populasi.

Dengan sumber daya, sebuah jadwal harus *resource-feasible*. Makespan dihitung lewat
**Schedule Generation Scheme (SGS)** yang menghormati precedence **dan** kapasitas:

- **Serial SGS** dengan urutan prioritas **LFT** (latest finish time — task paling
  genting dijadwalkan lebih dulu). Tiap task ditempatkan di waktu terdini yang tidak
  melanggar kapasitas selama seluruh durasinya.
- **Parallel SGS** (penjadwalan berbasis waktu) sebagai cadangan.

Diambil makespan **terkecil** di antara keduanya. Sifat penting:

```
makespan_resource(d) ≥ makespan_CPM(d)        (sumber daya hanya menunda)
```

Optimasi yang dipakai:

- **Short-circuit.** Bila serial-LFT sudah menyentuh batas bawah CPM, jadwal itu pasti
  optimal untuk `d` tersebut, sehingga parallel SGS dilewati.
- **Memoization.** Hasil jadwal di-cache per vektor durasi integer (`tuple(d)`), berguna
  saat plateau/clustering menghasilkan banyak `d` identik.
- **CPM pre-filter.** Karena `makespan_resource ≥ makespan_CPM`, titik dengan
  `makespan_CPM > deadline` pasti infeasible dan tidak perlu SGS.

> **Catatan.** SGS bersifat heuristik — memberi *batas atas* makespan resource-feasible
> minimum. Pada instans ini, hasilnya diverifikasi **persis sama** dengan RCPSP eksak
> (CP-SAT): gap 0 pada 60 vektor durasi acak. Jadi SOAC tidak salah menolak optimum sejati.

### Fitness

Bentuk fitness tidak berubah; hanya `v` yang kini memakai makespan resource-feasible:

```
Z_norm = Z / Z_max,   Z_max = Σ_j c_j (d_max_j − d_min_j)
v      = max(0, makespan_resource − T_deadline)

F(d) = 1 / (1 + Z_norm)            jika feasible   →  F ∈ [½, 1]
F(d) = 1 / (2 + Z_norm + v)        jika infeasible →  F <  ½
```

Semua vektor durasi feasible berbiaya minimum membentuk **plateau** F maksimum —
struktur multimodal yang selaras dengan gagasan "banyak akar" pada kasus Diophantine.

### Pada ILP ground truth

`ilp_ground_truth` bercabang otomatis (`scipy.optimize.milp` / HiGHS):

- **Tanpa sumber daya:** LP start-kontinu ringan (`s_j` real, precedence sebagai
  `s_j ≥ s_p + d_p`). Cukup karena tanpa kapasitas masalahnya linier.
- **Dengan sumber daya:** formulasi **time-indexed multi-mode**. Biner `u[j,t,δ] = 1`
  bila task `j` mulai pada waktu `t` dengan durasi `δ`. Kapasitas ditegakkan sebagai
  kendala *cumulative* per (sumber daya, waktu). Ini ILP eksak. Ruang variabel
  dipersempit dengan jendela start `[ES_j, LS_j]` dari CPM.

---

## Cara menjalankan

Dependensi: `pysne` (branch `eksperimen-int`), `numpy`, `scipy ≥ 1.9`.

```bash
pip install "git+https://github.com/p2ms-optimization/pysne.git@eksperimen-int"
python run_crashing_adiel_new.py
```

Ketiga file JSON harus berada di folder yang sama dengan `run_crashing_adiel_new.py`
(path dibaca relatif via `Path(__file__).resolve().parent`).

Untuk versi Colab satu-file (tanpa layout terpisah), lihat notebook
`SOAC_project_crashing_colab.ipynb`.

---

## Parameter dan waktu komputasi

`PARAMS` di `run_crashing_adiel_new.py` mengatur skala clustering dan SDOA. Karena
**setiap evaluasi fitness memanggil SGS** (bukan CPM tervektorisasi), biaya per titik
naik ratusan kali lipat dibanding versi tanpa sumber daya:

- CPM (tervektorisasi): ~0.001 ms/titik
- SGS resource-feasible: ~0.75 ms/titik

Konsekuensinya:

- Setelan lebih hemat (mis. `m_cluster=4096`, `sdoa_m=512`) → lebih cepat, cakupan
  plateau lebih sempit.
- Setelan lebih besar (mis. `m_cluster=16384`, `sdoa_m=2048`) → lebih lambat, cakupan
  lebih lengkap.

Bila ingin lebih cepat tanpa mengubah cakupan, langkah lanjutan yang mungkin: batasi
pengecekan SGS hanya ke sumber daya yang berpotensi mengikat (mayoritas dari 24 sumber
daya kapasitasnya tak pernah tercapai).

---

## Validasi (data 25 task, T = 241)

- **ILP ground truth (RCPSP-TCT):** `Z* = 260`, makespan 241, resource-feasible.
- **SOAC:** mencapai `Z = 260` (gap 0 terhadap ILP).
- **Scheduler SGS:** cocok persis dengan RCPSP eksak CP-SAT (gap makespan 0 pada 60
  vektor acak).
- **Jadwal resource-feasible:** berbeda dari CPM murni karena leveling — mis. *Concrete
  Slabs* bergeser dari s=47 ke s=77 untuk menghindari bentrok *G.C. Labor Crew*
  (permintaan 4 > kapasitas 3).
- **Himpunan optimum:** 3 solusi, seluruhnya `Z = 260`, berbeda hanya pada dua task
  berbiaya sama (`c = 40`): *Plumbing Trim* dan *Cleaning*, dengan
  `(d_PlumbingTrim, d_Cleaning) ∈ {(5,12), (4,13), (3,14)}`.

---

## Batasan yang perlu dicatat

1. **Kapasitas inert pada T = 241.** Pada instans/deadline ini, kapasitas tidak
   mengubah jawaban: `Z*` tetap 260 dan ketiga optimum sama persis dengan model CPM,
   karena konflik sumber daya bisa di-*level* dalam slack tanpa menambah biaya atau
   memperpanjang makespan. Kapasitas berpotensi mengikat (dan menaikkan `Z*`) pada
   deadline yang lebih ketat — kandidat pengujian yang baik untuk laporan.

2. **Sel tengah plateau bisa terlewat.** SOAC murni cenderung menangkap **ujung-ujung**
   plateau kolinear dan dapat melewatkan sel tengah — mis. `(4,13)` di antara `(5,12)`
   dan `(3,14)`. Ini keterbatasan cakupan clustering pada plateau tanpa lembah pemisah,
   **bukan** disebabkan (maupun diperbaiki oleh) kapasitas.

3. **SGS adalah batas atas heuristik.** Terverifikasi eksak pada instans ini, tetapi
   secara umum tidak dijamin optimal untuk sembarang `d`. Ground truth yang eksak tetap
   ILP time-indexed.

4. **Performa.** Lihat bagian "Parameter dan waktu komputasi".

---

## Kompatibilitas mundur

Bila `resource_capacity` dan `resource_requirements` tidak diberikan, kelas otomatis
kembali ke perilaku **CPM murni** (makespan forward-pass tervektorisasi, ILP
start-kontinu). Nama masalah menandai mode aktif: `ProjectCrashing[RCPSP](...)` versus
`ProjectCrashing[CPM](...)`.

---

## Rujukan

- Sidarto, K. A., Kania, A., Sumarti, N. (2017). *Spiral Optimization Algorithm with
  Clustering*.
- Sumarti, N., dkk. (2023). *SOAC untuk penyelesaian integer/Diophantine*.
