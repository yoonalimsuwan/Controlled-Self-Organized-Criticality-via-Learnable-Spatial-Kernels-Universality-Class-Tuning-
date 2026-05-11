import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time

# Initial setup
cp.random.seed(42)
np.random.seed(42)
os.makedirs("results", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def mle_tau(avalanches, xmin_pct=10):
    """
    Maximum Likelihood Estimator for power-law exponent τ.
    Runs on CPU (numpy) because the avalanches list is 1D and small.
    """
    arr = np.array(avalanches, dtype=float)
    arr = arr[arr >= 1]
    if len(arr) < 20:
        return None
    xmin = max(1.0, np.percentile(arr, xmin_pct))
    arr  = arr[arr >= xmin]
    if len(arr) < 5:
        return None
    return float(1.0 + len(arr) / np.sum(np.log(arr / xmin)))

def sigma_wave(S, S_c, L, n_events=80):
    """
    Wave branching ratio σ = ⟨|wave_{k+1}| / |wave_k|⟩
    OPTIMIZATION: Replaced Python sets and coordinate mappings with 
    pure GPU boolean masks to prevent CPU-GPU synchronization bottlenecks.
    """
    ratios = []
    S_run  = S.copy()
    
    for _ in range(n_events):
        x, y = cp.random.randint(1, L-1, 2).tolist()
        S_run[x, y] += 1
        S_w  = S_run.copy()
        
        seen_mask = cp.zeros((L, L), dtype=bool)
        cur_mask  = (S_w >= S_c)
        
        while cur_mask.any():
            S_n = S_w.copy()
            tp = cur_mask.astype(cp.float32)
            
            # Topple only the current wave front
            S_n -= 4 * tp
            S_n += (cp.roll(tp, 1, 0) + cp.roll(tp, -1, 0) +
                    cp.roll(tp, 1, 1) + cp.roll(tp, -1, 1))
            
            S_n[0, :]  = cp.clip(S_n[0, :],  0, S_c-1)
            S_n[-1, :] = cp.clip(S_n[-1, :], 0, S_c-1)
            S_n[:, 0]  = cp.clip(S_n[:, 0],  0, S_c-1)
            S_n[:, -1] = cp.clip(S_n[:, -1], 0, S_c-1)
            
            seen_mask |= cur_mask
            nxt_mask = (S_n >= S_c) & ~seen_mask
            
            cur_sum = float(cur_mask.sum())
            nxt_sum = float(nxt_mask.sum())
            
            if cur_sum > 0 and nxt_sum > 0:
                ratios.append(nxt_sum / cur_sum)
                
            cur_mask = nxt_mask
            S_w = S_n
            
        S_run = S_w
        
    return float(np.mean(ratios)) if ratios else 0.0

def fixed_point_error(S, S_c, L, a1=0.0003, a2=0.0002, a3=0.00005, a4=0.0003):
    """
    ||u(S) + Ψ[S] - S|| / ||S||
    Fully executed on GPU.
    """
    lap = (cp.roll(S, 1, 0) + cp.roll(S, -1, 0) +
           cp.roll(S, 1, 1) + cp.roll(S, -1, 1) - 4 * S)
    dx  = cp.roll(S, -1, 1) - cp.roll(S, 1, 1)
    dy  = cp.roll(S, -1, 0) - cp.roll(S, 1, 0)
    
    u   = a1 * cp.mean(S) + a2 * lap + a3 * (dx**2 + dy**2) / 4 - a4 * S
    S1  = S + u
    
    # BTW topple
    S1_r = S1.copy()
    for _ in range(500):
        mask = S1_r >= S_c
        if not mask.any(): 
            break
        tp   = mask.astype(cp.float32)
        S1_r -= 4 * tp
        S1_r += (cp.roll(tp, 1, 0) + cp.roll(tp, -1, 0) +
                 cp.roll(tp, 1, 1) + cp.roll(tp, -1, 1))
        
        S1_r[0, :]  = cp.clip(S1_r[0, :],  0, S_c-1)
        S1_r[-1, :] = cp.clip(S1_r[-1, :], 0, S_c-1)
        S1_r[:, 0]  = cp.clip(S1_r[:, 0],  0, S_c-1)
        S1_r[:, -1] = cp.clip(S1_r[:, -1], 0, S_c-1)
        
    return float(cp.linalg.norm(S1_r - S) / (cp.linalg.norm(S) + 1e-10))


# ══════════════════════════════════════════════════════════════════
# SECTION 248 — Deterministic SSC-Controlled SOC
# ══════════════════════════════════════════════════════════════════

class Section248:
    """
    SSC-248++ Deterministic Controller (GPU Accelerated)
    """
    A1, A2, A3, A4 = 0.0003, 0.0002, 0.00005, 0.0003

    def __init__(self, L=512, S_c=4):
        self.L   = L
        self.S_c = S_c
        # Initialize directly on GPU
        self.S   = cp.random.randint(2, S_c, (L, L)).astype(cp.float32)

    def _control_u(self):
        S   = self.S
        lap = (cp.roll(S, 1, 0) + cp.roll(S, -1, 0) +
               cp.roll(S, 1, 1) + cp.roll(S, -1, 1) - 4 * S)
        dx  = cp.roll(S, -1, 1) - cp.roll(S, 1, 1)
        dy  = cp.roll(S, -1, 0) - cp.roll(S, 1, 0)
        return (self.A1 * cp.mean(S) + self.A2 * lap
                + self.A3 * (dx**2 + dy**2) / 4 - self.A4 * S)

    def _btw_topple(self):
        A = 0
        while True:
            mask = self.S >= self.S_c
            if not mask.any(): 
                break
            tp     = mask.astype(cp.float32)
            self.S -= 4 * tp
            self.S += (cp.roll(tp, 1, 0) + cp.roll(tp, -1, 0) +
                       cp.roll(tp, 1, 1) + cp.roll(tp, -1, 1))
            
            self.S[0, :]  = cp.clip(self.S[0, :],  0, self.S_c-1)
            self.S[-1, :] = cp.clip(self.S[-1, :], 0, self.S_c-1)
            self.S[:, 0]  = cp.clip(self.S[:, 0],  0, self.S_c-1)
            self.S[:, -1] = cp.clip(self.S[:, -1], 0, self.S_c-1)
            A += int(mask.sum())
        return A

    def run(self, n_steps=8000, warmup=2000, measure_every=500):
        avalanches = []
        history    = {'tau':[], 'sigma':[], 'fp_err':[], 'step':[]}

        print(f"  Section 248: L={self.L}, steps={n_steps}")
        for t in range(n_steps + warmup):
            self.S += self._control_u()
            x, y = cp.random.randint(1, self.L-1, 2).tolist()
            self.S[x, y] += 1
            self.S  = cp.clip(self.S, 0, None)
            
            A = self._btw_topple()
            if t >= warmup and A > 0:
                avalanches.append(A)

            if (t >= warmup and t % measure_every == 0
                    and len(avalanches) > 50):
                tau = mle_tau(avalanches[-500:])
                sig = sigma_wave(self.S, self.S_c, self.L, n_events=80)
                fpe = fixed_point_error(self.S, self.S_c, self.L,
                                        self.A1, self.A2,
                                        self.A3, self.A4)
                if tau:
                    history['tau'].append(tau)
                    history['sigma'].append(sig)
                    history['fp_err'].append(fpe)
                    history['step'].append(t)
                    print(f"    step {t:5d} | τ={tau:.4f} | "
                          f"σ={sig:.4f} | FP_err={fpe:.5f}")

        return avalanches, history

    def plot(self, avalanches, history, btw_tau=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Section 248++: Deterministic SSC-Controlled SOC (GPU Accelerated)\n"
            "S(x,t+1) = S + a₁⟨S⟩ + a₂∇²S + a₃|∇S|² − a₄S + Ψ[S]",
            fontsize=11, fontweight='bold')

        ax = axes[0]
        ax.plot(history['step'], history['tau'],
                color='#E91E63', lw=2, label='τ (SSC-248)')
        ax.axhline(1.3, color='red', ls='--', lw=1.5, label='τ*=1.3')
        ax.axhspan(1.25, 1.35, alpha=0.1, color='red')
        if btw_tau:
            ax.axhline(btw_tau, color='blue', ls=':', lw=1.5,
                       label=f'BTW τ={btw_tau:.3f}')
        ax.set_xlabel('Step'); ax.set_ylabel('τ (MLE)')
        ax.set_title('τ convergence\n(target: [1.25,1.35])')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

        ax = axes[1]
        ax.plot(history['step'], history['sigma'],
                color='#FF9800', lw=2, label='σ (wave)')
        ax.axhline(1.0, color='red', ls='--', lw=1.5, label='σ*=1')
        ax.axhspan(0.8, 1.2, alpha=0.08, color='red')
        ax.set_xlabel('Step'); ax.set_ylabel('σ')
        ax.set_title('Branching ratio σ → 1\n(marginal criticality)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

        ax = axes[2]
        ax.semilogy(history['step'], history['fp_err'],
                    color='purple', lw=2,
                    label='||u(S)+Ψ[S]-S||/||S||')
        ax.axhline(0.05, color='red', ls='--', alpha=0.5,
                   label='0.05 threshold')
        ax.set_xlabel('Step'); ax.set_ylabel('FP error (log)')
        ax.set_title('Fixed point condition\nu(S*)+Ψ[S*]=S* → 0')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig('results/sec248_result_gpu.png', dpi=150, bbox_inches='tight')
        print("  Saved: results/sec248_result_gpu.png")
        plt.show()

def run_section_248():
    start_time = time.time()
    print("\n" + "="*60)
    print("SECTION 248 — Deterministic SSC-Controlled SOC [GPU MODE]")
    print("="*60)

    L = 512
    
    print(f"  Running BTW baseline (L={L}) on GPU...")
    # Initialize directly on GPU
    S_b = cp.random.randint(2, 4, (L, L)).astype(cp.float32)
    avs_b = []
    
    total_steps = 10000 + 2000
    for t in range(total_steps):
        if t % 2000 == 0:
            print(f"    Baseline progress: {t}/{total_steps} steps...")
            
        x, y = cp.random.randint(0, L, 2).tolist()
        S_b[x, y] += 1
        
        while True:
            mask = S_b >= 4
            if not mask.any(): 
                break
            tp   = mask.astype(cp.float32)
            S_b -= 4 * tp
            S_b += (cp.roll(tp, 1, 0) + cp.roll(tp, -1, 0) +
                    cp.roll(tp, 1, 1) + cp.roll(tp, -1, 1))
            
            S_b[0, :]  = cp.clip(S_b[0, :],  0, 3)
            S_b[-1, :] = cp.clip(S_b[-1, :], 0, 3)
            S_b[:, 0]  = cp.clip(S_b[:, 0],  0, 3)
            S_b[:, -1] = cp.clip(S_b[:, -1], 0, 3)
            
        if t >= 2000:
            A = int(cp.sum(S_b >= 4))
            if A > 0: 
                avs_b.append(A)
            
    btw_tau = mle_tau(avs_b)
    print(f"  BTW: τ={btw_tau:.4f}" if btw_tau else "  BTW: Insufficient avalanches to calculate τ")

    # SSC-248++ on GPU
    model        = Section248(L=L, S_c=4)
    avs, history = model.run(n_steps=8000, warmup=2000)

    tau_f = history['tau'][-1]   if history['tau']   else None
    sig_f = history['sigma'][-1] if history['sigma'] else None
    fpe_f = history['fp_err'][-1]if history['fp_err']else None

    print(f"\n  Results (Completed in {time.time() - start_time:.2f} seconds):")
    print(f"    τ     = {tau_f:.4f}  (target [1.25,1.35]) " if tau_f else "    τ     = None",
          f"{'✓' if tau_f and 1.25<=tau_f<=1.35 else '✗'}")
    print(f"    σ     = {sig_f:.4f}  (target ~1.0) " if sig_f else "    σ     = None",
          f"{'✓' if sig_f and 0.8<=sig_f<=1.2 else '✗'}")
    print(f"    FP err= {fpe_f:.6f}  (target <0.05) " if fpe_f else "    FP err= None",
          f"{'✓' if fpe_f and fpe_f<0.05 else '✗'}")

    model.plot(avs, history, btw_tau=btw_tau)
    return tau_f, sig_f, fpe_f

# Execute the run
run_section_248()
