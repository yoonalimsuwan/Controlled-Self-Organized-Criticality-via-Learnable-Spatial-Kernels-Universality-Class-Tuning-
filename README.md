# SSC-SOC Controlled Criticality

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


==================================================


## Citation

```bibtex
@article{limsuwan2026ssc,
  title={Structural Calculus and Semantic-State Contraction:
         Toward Deterministic Multi-Scale Physical Computation},
  author={Limsuwan, Yoon A},
  year={2026}
}
```
