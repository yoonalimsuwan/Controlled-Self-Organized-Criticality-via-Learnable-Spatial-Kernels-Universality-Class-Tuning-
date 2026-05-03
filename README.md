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

## Citation

```bibtex
@article{limsuwan2026ssc,
  title={Structural Calculus and Semantic-State Contraction:
         Toward Deterministic Multi-Scale Physical Computation},
  author={Limsuwan, Yoon A},
  year={2026}
}
```
