# SSC-SOC Controlled Criticality

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)

Implementation of Structural Calculus and Semantic-State Contraction (SSC) applied to Self-Organized Criticality (SOC), from the paper:

> **Structural Calculus and Semantic-State Contraction: Toward Deterministic Multi-Scale Physical Computation**  
> Yoon A Limsuwan
> https://doi.org/10.5281/zenodo.19814975
>
> *A Rigorous Applied Framework for Resolving Seven Canonical Challenges via Deterministic Semantic-State and Structural Calculus**  
> Yoon A Limsuwan
> https://doi.org/10.5281/zenodo.19869633

## Sections Implemented

| Section | Title | Key Result |
|---------|-------|------------|
| 248 | Deterministic SSC-Controlled SOC | FP error < 0.003, σ→1, τ∈[1.25,1.35] |
| 252 | Learnable Kernel SOC | α controls τ, Δτ=0.48 |
| 398 | Empirical Fit τ(α) | R²=0.993, τ=τ_∞+A·exp(-kα) |

## Key Results

**Section 248** — Deterministic SSC controller:
- Fixed point error: < 0.003
- Branching ratio σ → 1 (marginal criticality)  
- Power-law exponent τ ∈ [1.25, 1.35]

**Section 252** — Learnable kernel K_θ(r) = (r+ε)^(-α)·exp(-r/λ):
- τ range: 1.317 – 1.799 (Δτ = 0.48)
- α → ∞: local BTW (τ ≈ 1.32)
- α → 2: long-range Lévy (τ ≈ 1.80)
- Monotone: τ(α₁) > τ(α₂) for α₁ < α₂ ✓

**Section 398** — Empirical fit:
- τ(α) = τ_∞ + A·exp(-kα)
- τ_∞ = 1.312, A = 1.242, k = 0.596
- R² = 0.993

## Install

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
# Run all sections
python ssc_soc_complete.py

# Run individual sections
python ssc_soc_complete.py --sec 248
python ssc_soc_complete.py --sec 252
python ssc_soc_complete.py --sec 398
```

Results saved to `./results/`

## Mathematical Framework

### Section 248: Control Law
```
S(x,t+1) = S(x,t) + u(x,t) + Ψ[S(x,t)]

u(x,t) = a₁⟨S⟩ + a₂∇²S + a₃|∇S|² - a₄S

Fixed point condition: u(S*) + Ψ[S*] = 0
→ a₁ = a₄ (global/local balance)
```

### Section 252: Kernel
```
K_θ(r) = (r+ε)^(-α) · exp(-r/λ)

α → ∞: K → δ(r=1)  [local BTW]
α → 2: K ~ r^(-2)  [long-range Lévy]
```

### Section 398: Empirical Fit
```
τ(α) = τ_∞ + A · exp(-k·α)

RG interpretation:
  k   = stability eigenvalue of β-function
  τ_∞ = BTW universality class fixed point
```

L = 300

Simulating alpha=1.5 ...
    [Warmup] Step 25000/150000 | Elapsed: 0.9 mins
    [Warmup] Step 50000/150000 | Elapsed: 2.4 mins
    [Warmup] Step 75000/150000 | Elapsed: 3.9 mins
    [Recording] Step 100000/150000 | Elapsed: 5.4 mins
    [Recording] Step 125000/150000 | Elapsed: 6.9 mins
  ↳ Result for alpha=1.5: τ = 1.9014 (Calculated from 18024 events, Time: 8.4 mins)

Simulating alpha=2.0 ...
    [Warmup] Step 25000/150000 | Elapsed: 1.0 mins
    [Warmup] Step 50000/150000 | Elapsed: 2.9 mins
    [Warmup] Step 75000/150000 | Elapsed: 4.8 mins
    [Recording] Step 100000/150000 | Elapsed: 6.8 mins
    [Recording] Step 125000/150000 | Elapsed: 8.9 mins
  ↳ Result for alpha=2.0: τ = 1.7145 (Calculated from 20151 events, Time: 10.9 mins)

Simulating alpha=3.0 ...
    [Warmup] Step 25000/150000 | Elapsed: 2.6 mins
    [Warmup] Step 50000/150000 | Elapsed: 8.3 mins
    [Warmup] Step 75000/150000 | Elapsed: 14.2 mins
    [Recording] Step 100000/150000 | Elapsed: 19.9 mins
    [Recording] Step 125000/150000 | Elapsed: 25.5 mins
  ↳ Result for alpha=3.0: τ = 1.4508 (Calculated from 26702 events, Time: 31.1 mins)

Simulating alpha=5.0 ...
    [Warmup] Step 25000/150000 | Elapsed: 3.09 mins
    [Warmup] Step 50000/150000 | Elapsed: 31.96 mins
    [Warmup] Step 75000/150000 | Elapsed: 60.58 mins
    [Recording] Step 100000/150000 | Elapsed: 88.65 mins
    [Recording] Step 125000/150000 | Elapsed: 116.58 mins
    [Auto-Save] Data saved to results/avalanches_alpha_5.0.csv
  ↳ Result for alpha=5.0: τ = 1.2420 (Events: 37616, Time: 145.17 mins)


Simulating alpha=10.0 ...
    [Warmup] Step 25000/150000 | Elapsed: 1.58 mins
    [Warmup] Step 50000/150000 | Elapsed: 45.19 mins
    [Warmup] Step 75000/150000 | Elapsed: 90.62 mins
    [Recording] Step 100000/150000 | Elapsed: 135.03 mins
    [Recording] Step 125000/150000 | Elapsed: 179.00 mins
    [Auto-Save] Data saved to results/avalanches_alpha_10.0.csv
  ↳ Result for alpha=10.0: τ = 1.2411 (Events: 40963, Time: 223.55 mins)

L = 512

Simulating alpha = 1.50
Starting new simulation
[Warmup] Step 10000/150000 | Elapsed = 0.13 mins
[Warmup] Step 20000/150000 | Elapsed = 0.28 mins
[Warmup] Step 30000/150000 | Elapsed = 0.43 mins
[Warmup] Step 40000/150000 | Elapsed = 0.59 mins
[Warmup] Step 50000/150000 | Elapsed = 0.74 mins
[Warmup] Step 60000/150000 | Elapsed = 0.97 mins
[Warmup] Step 70000/150000 | Elapsed = 1.24 mins
[Warmup] Step 80000/150000 | Elapsed = 1.54 mins
[Warmup] Step 90000/150000 | Elapsed = 1.83 mins
[Recording] Step 100000/150000 | Elapsed = 2.17 mins
[Recording] Step 110000/150000 | Elapsed = 2.49 mins
[Recording] Step 120000/150000 | Elapsed = 2.84 mins
[Recording] Step 130000/150000 | Elapsed = 3.17 mins
[Recording] Step 140000/150000 | Elapsed = 3.52 mins

------------------------------------------------------------
 alpha = 1.50 | tau = 1.91534
⏱ Runtime = 3.85 mins

Simulating alpha = 2.00
Starting new simulation
[Warmup] Step 10000/150000 | Elapsed = 0.14 mins
[Warmup] Step 20000/150000 | Elapsed = 0.29 mins
[Warmup] Step 30000/150000 | Elapsed = 0.44 mins
[Warmup] Step 40000/150000 | Elapsed = 0.60 mins
[Warmup] Step 50000/150000 | Elapsed = 0.79 mins
[Warmup] Step 60000/150000 | Elapsed = 1.09 mins
[Warmup] Step 70000/150000 | Elapsed = 1.48 mins
[Warmup] Step 80000/150000 | Elapsed = 1.88 mins
[Warmup] Step 90000/150000 | Elapsed = 2.31 mins
[Recording] Step 100000/150000 | Elapsed = 2.75 mins
[Recording] Step 110000/150000 | Elapsed = 3.18 mins
[Recording] Step 120000/150000 | Elapsed = 3.66 mins
[Recording] Step 130000/150000 | Elapsed = 4.11 mins
[Recording] Step 140000/150000 | Elapsed = 4.57 mins

------------------------------------------------------------
alpha = 2.00 | tau = 1.70724
⏱ Runtime = 5.04 mins

Simulating alpha=3.0 on Grid 512x512 ...
    Starting a new simulation...
    [Warmup] Step 10000/150000 | Elapsed (this run): 0.22 mins
    [Warmup] Step 20000/150000 | Elapsed (this run): 0.38 mins
    [Warmup] Step 30000/150000 | Elapsed (this run): 0.55 mins
    [Warmup] Step 40000/150000 | Elapsed (this run): 0.78 mins
    [Warmup] Step 50000/150000 | Elapsed (this run): 1.22 mins
    [Warmup] Step 60000/150000 | Elapsed (this run): 2.72 mins
    [Warmup] Step 70000/150000 | Elapsed (this run): 4.29 mins
    [Warmup] Step 80000/150000 | Elapsed (this run): 5.84 mins
    [Warmup] Step 90000/150000 | Elapsed (this run): 7.39 mins
    [Recording] Step 100000/150000 | Elapsed (this run): 8.94 mins
    [Recording] Step 110000/150000 | Elapsed (this run): 10.51 mins
    [Recording] Step 120000/150000 | Elapsed (this run): 12.10 mins
    [Recording] Step 130000/150000 | Elapsed (this run): 13.61 mins
    [Recording] Step 140000/150000 | Elapsed (this run): 15.17 mins
  ↳ Result for alpha=3.0: τ = 1.4409 (Run Time: 16.77 mins)

Simulating alpha=5.0 on Grid 512x512 ...
    [Warmup] Step 5000/150000 | Elapsed: 0.15 mins
    [Warmup] Step 10000/150000 | Elapsed: 0.22 mins
    [Warmup] Step 15000/150000 | Elapsed: 0.30 mins
    [Warmup] Step 20000/150000 | Elapsed: 0.40 mins
    [Warmup] Step 25000/150000 | Elapsed: 0.50 mins
    [Warmup] Step 30000/150000 | Elapsed: 0.62 mins
    [Warmup] Step 35000/150000 | Elapsed: 0.76 mins
    [Warmup] Step 40000/150000 | Elapsed: 0.91 mins
    [Warmup] Step 45000/150000 | Elapsed: 1.09 mins
    [Warmup] Step 50000/150000 | Elapsed: 1.32 mins
    [Warmup] Step 55000/150000 | Elapsed: 1.61 mins
    [Warmup] Step 60000/150000 | Elapsed: 2.02 mins
    [Warmup] Step 65000/150000 | Elapsed: 2.66 mins
    [Warmup] Step 70000/150000 | Elapsed: 5.97 mins
    [Warmup] Step 75000/150000 | Elapsed: 16.37 mins
    [Warmup] Step 80000/150000 | Elapsed: 27.87 mins
    [Warmup] Step 85000/150000 | Elapsed: 38.47 mins
    [Warmup] Step 90000/150000 | Elapsed: 49.56 mins
    [Warmup] Step 95000/150000 | Elapsed: 60.70 mins
    [Recording] Step 100000/150000 | Elapsed: 71.41 mins
    [Recording] Step 105000/150000 | Elapsed: 82.28 mins
    [Recording] Step 110000/150000 | Elapsed: 93.32 mins
    [Recording] Step 115000/150000 | Elapsed: 103.95 mins
    [Recording] Step 120000/150000 | Elapsed: 115.13 mins
    [Recording] Step 125000/150000 | Elapsed: 126.58 mins
    [Recording] Step 130000/150000 | Elapsed: 137.75 mins
    [Recording] Step 135000/150000 | Elapsed: 149.07 mins
    [Recording] Step 140000/150000 | Elapsed: 160.13 mins
    [Recording] Step 145000/150000 | Elapsed: 171.27 mins
    [Auto-Save] Data saved to results/avalanches_alpha_5.0.csv
  ↳ Result for alpha=5.0: τ = 1.2239 (Events: 37943, Time: 182.16 mins)

3D L = 64

▶ Simulating alpha = 1.0
▶ Grid = 64 x 64 x 64
[Warmup] Step 5000/150000 | Elapsed = 0.12 mins
[Warmup] Step 10000/150000 | Elapsed = 0.22 mins
[Warmup] Step 15000/150000 | Elapsed = 0.33 mins
[Warmup] Step 20000/150000 | Elapsed = 0.44 mins
[Warmup] Step 25000/150000 | Elapsed = 0.56 mins
[Warmup] Step 30000/150000 | Elapsed = 0.67 mins
[Warmup] Step 35000/150000 | Elapsed = 0.79 mins
[Warmup] Step 40000/150000 | Elapsed = 0.91 mins
[Warmup] Step 45000/150000 | Elapsed = 1.03 mins
[Warmup] Step 50000/150000 | Elapsed = 1.15 mins
[Warmup] Step 55000/150000 | Elapsed = 1.27 mins
[Warmup] Step 60000/150000 | Elapsed = 1.40 mins
[Warmup] Step 65000/150000 | Elapsed = 1.54 mins
[Warmup] Step 70000/150000 | Elapsed = 1.68 mins
[Warmup] Step 75000/150000 | Elapsed = 1.83 mins
[Warmup] Step 80000/150000 | Elapsed = 2.00 mins
[Warmup] Step 85000/150000 | Elapsed = 2.17 mins
[Warmup] Step 90000/150000 | Elapsed = 2.35 mins
[Warmup] Step 95000/150000 | Elapsed = 2.54 mins
[Recording] Step 100000/150000 | Elapsed = 2.74 mins
[Recording] Step 105000/150000 | Elapsed = 2.94 mins
[Recording] Step 110000/150000 | Elapsed = 3.16 mins
[Recording] Step 115000/150000 | Elapsed = 3.37 mins
[Recording] Step 120000/150000 | Elapsed = 3.58 mins
[Recording] Step 125000/150000 | Elapsed = 3.79 mins
[Recording] Step 130000/150000 | Elapsed = 3.98 mins
[Recording] Step 135000/150000 | Elapsed = 4.17 mins
[Recording] Step 140000/150000 | Elapsed = 4.36 mins
[Recording] Step 145000/150000 | Elapsed = 4.54 mins

✅ alpha = 1.0 | tau = 2.51615
📈 events = 12566
⏱ runtime = 4.72 mins

▶ Simulating alpha = 2.0
▶ Grid = 64 x 64 x 64
[Warmup] Step 5000/150000 | Elapsed = 0.10 mins
[Warmup] Step 10000/150000 | Elapsed = 0.21 mins
[Warmup] Step 15000/150000 | Elapsed = 0.32 mins
[Warmup] Step 20000/150000 | Elapsed = 0.43 mins
[Warmup] Step 25000/150000 | Elapsed = 0.55 mins
[Warmup] Step 30000/150000 | Elapsed = 0.66 mins
[Warmup] Step 35000/150000 | Elapsed = 0.79 mins
[Warmup] Step 40000/150000 | Elapsed = 0.91 mins
[Warmup] Step 45000/150000 | Elapsed = 1.03 mins
[Warmup] Step 50000/150000 | Elapsed = 1.15 mins
[Warmup] Step 55000/150000 | Elapsed = 1.29 mins
[Warmup] Step 60000/150000 | Elapsed = 1.47 mins
[Warmup] Step 65000/150000 | Elapsed = 1.68 mins
[Warmup] Step 70000/150000 | Elapsed = 2.00 mins
[Warmup] Step 75000/150000 | Elapsed = 2.27 mins
[Warmup] Step 80000/150000 | Elapsed = 2.51 mins
[Warmup] Step 85000/150000 | Elapsed = 2.73 mins
[Warmup] Step 90000/150000 | Elapsed = 2.94 mins
[Warmup] Step 95000/150000 | Elapsed = 3.16 mins
[Recording] Step 100000/150000 | Elapsed = 3.38 mins
[Recording] Step 105000/150000 | Elapsed = 3.60 mins
[Recording] Step 110000/150000 | Elapsed = 3.84 mins
[Recording] Step 115000/150000 | Elapsed = 4.08 mins
[Recording] Step 120000/150000 | Elapsed = 4.31 mins
[Recording] Step 125000/150000 | Elapsed = 4.53 mins
[Recording] Step 130000/150000 | Elapsed = 4.75 mins
[Recording] Step 135000/150000 | Elapsed = 4.96 mins
[Recording] Step 140000/150000 | Elapsed = 5.17 mins
[Recording] Step 145000/150000 | Elapsed = 5.40 mins
💾 Saved: results/avalanches_3d_alpha_2.0.csv

✅ alpha = 2.0 | tau = 2.26340
📈 events = 14581
⏱ runtime = 5.63 mins

▶ Simulating alpha = 3.0
▶ Grid = 64 x 64 x 64
[Warmup] Step 5000/150000 | Elapsed = 0.11 mins
[Warmup] Step 10000/150000 | Elapsed = 0.21 mins
[Warmup] Step 15000/150000 | Elapsed = 0.32 mins
[Warmup] Step 20000/150000 | Elapsed = 0.44 mins
[Warmup] Step 25000/150000 | Elapsed = 0.55 mins
[Warmup] Step 30000/150000 | Elapsed = 0.67 mins
[Warmup] Step 35000/150000 | Elapsed = 0.80 mins
[Warmup] Step 40000/150000 | Elapsed = 0.93 mins
[Warmup] Step 45000/150000 | Elapsed = 1.07 mins
[Warmup] Step 50000/150000 | Elapsed = 1.25 mins
[Warmup] Step 55000/150000 | Elapsed = 1.50 mins
[Warmup] Step 60000/150000 | Elapsed = 1.81 mins
[Warmup] Step 65000/150000 | Elapsed = 2.11 mins
[Warmup] Step 70000/150000 | Elapsed = 2.44 mins
[Warmup] Step 75000/150000 | Elapsed = 2.76 mins
[Warmup] Step 80000/150000 | Elapsed = 3.09 mins
[Warmup] Step 85000/150000 | Elapsed = 3.41 mins
[Warmup] Step 90000/150000 | Elapsed = 3.74 mins
[Warmup] Step 95000/150000 | Elapsed = 4.06 mins
[Recording] Step 100000/150000 | Elapsed = 4.40 mins
[Recording] Step 105000/150000 | Elapsed = 4.73 mins
[Recording] Step 110000/150000 | Elapsed = 5.07 mins
[Recording] Step 115000/150000 | Elapsed = 5.39 mins
[Recording] Step 120000/150000 | Elapsed = 5.73 mins
[Recording] Step 125000/150000 | Elapsed = 6.06 mins
[Recording] Step 130000/150000 | Elapsed = 6.40 mins
[Recording] Step 135000/150000 | Elapsed = 6.74 mins
[Recording] Step 140000/150000 | Elapsed = 7.07 mins
[Recording] Step 145000/150000 | Elapsed = 7.41 mins

✅ alpha = 3.0 | tau = 1.82857
📈 events = 18551
⏱ runtime = 7.74 mins

▶ Simulating alpha = 4.0
▶ Grid = 64 x 64 x 64
[Warmup] Step 5000/150000 | Elapsed = 0.11 mins
[Warmup] Step 10000/150000 | Elapsed = 0.22 mins
[Warmup] Step 15000/150000 | Elapsed = 0.33 mins
[Warmup] Step 20000/150000 | Elapsed = 0.45 mins
[Warmup] Step 25000/150000 | Elapsed = 0.57 mins
[Warmup] Step 30000/150000 | Elapsed = 0.70 mins
[Warmup] Step 35000/150000 | Elapsed = 0.85 mins
[Warmup] Step 40000/150000 | Elapsed = 1.03 mins
[Warmup] Step 45000/150000 | Elapsed = 1.29 mins
[Warmup] Step 50000/150000 | Elapsed = 1.78 mins
[Warmup] Step 55000/150000 | Elapsed = 2.27 mins
[Warmup] Step 60000/150000 | Elapsed = 2.82 mins
[Warmup] Step 65000/150000 | Elapsed = 3.39 mins
[Warmup] Step 70000/150000 | Elapsed = 3.92 mins
[Warmup] Step 75000/150000 | Elapsed = 4.47 mins
[Warmup] Step 80000/150000 | Elapsed = 5.00 mins
[Warmup] Step 85000/150000 | Elapsed = 5.53 mins
[Warmup] Step 90000/150000 | Elapsed = 6.08 mins
[Warmup] Step 95000/150000 | Elapsed = 6.64 mins
[Recording] Step 100000/150000 | Elapsed = 7.20 mins
[Recording] Step 105000/150000 | Elapsed = 7.77 mins
[Recording] Step 110000/150000 | Elapsed = 8.33 mins
[Recording] Step 115000/150000 | Elapsed = 8.88 mins
[Recording] Step 120000/150000 | Elapsed = 9.44 mins
[Recording] Step 125000/150000 | Elapsed = 10.00 mins
[Recording] Step 130000/150000 | Elapsed = 10.55 mins
[Recording] Step 135000/150000 | Elapsed = 11.10 mins
[Recording] Step 140000/150000 | Elapsed = 11.66 mins
[Recording] Step 145000/150000 | Elapsed = 12.21 mins

✅ alpha = 4.0 | tau = 1.62176
📈 events = 22171
⏱ runtime = 12.76 mins

▶ Running alpha = 5.0
[Warmup] Step 5000/150000 | Elapsed = 0.11 mins
[Warmup] Step 10000/150000 | Elapsed = 0.22 mins
[Warmup] Step 15000/150000 | Elapsed = 0.34 mins
[Warmup] Step 20000/150000 | Elapsed = 0.46 mins
[Warmup] Step 25000/150000 | Elapsed = 0.60 mins
[Warmup] Step 30000/150000 | Elapsed = 0.76 mins
[Warmup] Step 35000/150000 | Elapsed = 0.94 mins
[Warmup] Step 40000/150000 | Elapsed = 1.18 mins
[Warmup] Step 45000/150000 | Elapsed = 1.57 mins
[Warmup] Step 50000/150000 | Elapsed = 2.43 mins
[Warmup] Step 55000/150000 | Elapsed = 3.41 mins
[Warmup] Step 60000/150000 | Elapsed = 4.31 mins
[Warmup] Step 65000/150000 | Elapsed = 5.21 mins
[Warmup] Step 70000/150000 | Elapsed = 6.10 mins
[Warmup] Step 75000/150000 | Elapsed = 7.04 mins
[Warmup] Step 80000/150000 | Elapsed = 7.93 mins
[Warmup] Step 85000/150000 | Elapsed = 8.89 mins
[Warmup] Step 90000/150000 | Elapsed = 9.81 mins
[Warmup] Step 95000/150000 | Elapsed = 10.74 mins
[Recording] Step 100000/150000 | Elapsed = 11.64 mins
[Recording] Step 105000/150000 | Elapsed = 12.53 mins
[Recording] Step 110000/150000 | Elapsed = 13.46 mins
[Recording] Step 115000/150000 | Elapsed = 14.35 mins
[Recording] Step 120000/150000 | Elapsed = 15.27 mins
[Recording] Step 125000/150000 | Elapsed = 16.16 mins
[Recording] Step 130000/150000 | Elapsed = 17.06 mins
[Recording] Step 135000/150000 | Elapsed = 17.93 mins
[Recording] Step 140000/150000 | Elapsed = 18.84 mins
[Recording] Step 145000/150000 | Elapsed = 19.75 mins

✅ alpha = 5.0
✅ tau = 1.51676
✅ avalanches = 25635
⏱ Runtime = 20.67 mins

▶ Simulating alpha=10.0 on 3D Grid 64x64x64 ...
    [Warmup] Step 5000/150000 | Elapsed: 0.22 mins
    [Warmup] Step 10000/150000 | Elapsed: 0.35 mins
    [Warmup] Step 15000/150000 | Elapsed: 0.48 mins
    [Warmup] Step 20000/150000 | Elapsed: 0.63 mins
    [Warmup] Step 25000/150000 | Elapsed: 0.79 mins
    [Warmup] Step 30000/150000 | Elapsed: 0.98 mins
    [Warmup] Step 35000/150000 | Elapsed: 1.20 mins
    [Warmup] Step 40000/150000 | Elapsed: 1.46 mins
    [Warmup] Step 45000/150000 | Elapsed: 1.79 mins
    [Warmup] Step 50000/150000 | Elapsed: 2.24 mins
    [Warmup] Step 55000/150000 | Elapsed: 2.95 mins
    [Warmup] Step 60000/150000 | Elapsed: 4.55 mins
    [Warmup] Step 65000/150000 | Elapsed: 6.46 mins
    [Warmup] Step 70000/150000 | Elapsed: 8.32 mins
    [Warmup] Step 75000/150000 | Elapsed: 10.34 mins
    [Warmup] Step 80000/150000 | Elapsed: 12.30 mins
    [Warmup] Step 85000/150000 | Elapsed: 14.12 mins
    [Warmup] Step 90000/150000 | Elapsed: 15.99 mins
    [Warmup] Step 95000/150000 | Elapsed: 17.92 mins
    [Recording] Step 100000/150000 | Elapsed: 19.83 mins
    [Recording] Step 105000/150000 | Elapsed: 21.73 mins
    [Recording] Step 110000/150000 | Elapsed: 23.68 mins
    [Recording] Step 115000/150000 | Elapsed: 25.60 mins
    [Recording] Step 120000/150000 | Elapsed: 27.46 mins
    [Recording] Step 125000/150000 | Elapsed: 29.35 mins
    [Recording] Step 130000/150000 | Elapsed: 31.25 mins
    [Recording] Step 135000/150000 | Elapsed: 33.18 mins
    [Recording] Step 140000/150000 | Elapsed: 34.99 mins
    [Recording] Step 145000/150000 | Elapsed: 36.80 mins
    [Auto-Save] Data saved to results/avalanches_3d_alpha_10.0.csv
  ↳ Result for alpha=10.0: τ = 1.4003 (Events: 31495, Time: 38.76 mins)


## Citation

```bibtex
@article{limsuwan2026ssc,
  title={Structural Calculus and Semantic-State Contraction:
         Toward Deterministic Multi-Scale Physical Computation},
  author={Limsuwan, Yoon A},
  year={2026}
}
```
