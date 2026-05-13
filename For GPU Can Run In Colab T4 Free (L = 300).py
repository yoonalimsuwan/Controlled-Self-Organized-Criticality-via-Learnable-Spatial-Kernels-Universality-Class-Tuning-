import numpy as np
import cupy as cp
from cupyx.scipy.fft import next_fast_len
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import time

np.random.seed(42)
cp.random.seed(42)
os.makedirs("results", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def mle_tau(avalanches, xmin_pct=15):
    """
    Estimates the power-law exponent (tau) using Maximum Likelihood Estimation.
    """
    arr = np.array(avalanches, dtype=float)
    arr = arr[arr >= 1]
    if len(arr) < 50: return None
    xmin = max(5.0, np.percentile(arr, xmin_pct))
    arr = arr[arr >= xmin]
    if len(arr) < 20: return None
    return float(1.0 + len(arr) / np.sum(np.log(arr / xmin)))

# ══════════════════════════════════════════════════════════════════
# CUDA C++ KERNEL (Maximize GPU throughput, minimize Python overhead)
# ══════════════════════════════════════════════════════════════════

# Multi-Toppling Algorithm: Processes all cells exceeding the threshold in a single pass.
get_topples = cp.ElementwiseKernel(
    'float32 S_in', 
    'float32 tp, float32 S_out',
    '''
    if (S_in >= 1.0f) {
        tp = floor(S_in);
        S_out = S_in - tp;
    } else {
        tp = 0.0f;
        S_out = S_in;
    }
    ''',
    'get_topples'
)

# ══════════════════════════════════════════════════════════════════
# SECTION 252 — EXTREME GPU ACCELERATION (L=300)
# ══════════════════════════════════════════════════════════════════

def run_section_252_gpu_single(L=300):
    print("\n" + "="*65)
    print(f"SECTION 252 — EXTREME GPU (L={L}) : ABELIAN MULTI-TOPPLING")
    print("="*65)
    print("Note: Powered by C++ CUDA Elementwise + Zero-Allocation FFT.\n")

    alphas = [5.0] 
    sim_results = []

    WARMUP = 100000
    STEPS = 50000

    for alpha in alphas:
        start_time = time.time()
        print(f"▶ Simulating alpha={alpha:.1f} ...")

        # 1. Generate Kernel on CPU
        y, x = np.ogrid[-L//2+1:L//2+1, -L//2+1:L//2+1]
        r = np.sqrt(x**2 + y**2) + 1e-4
        cutoff = L / 4.0
        K_np = (r**-alpha) * np.exp(-r/cutoff)
        K_np[L//2-1, L//2-1] = 0.0
        K_np /= K_np.sum() 

        # 2. Transfer data to GPU (T4 or better recommended)
        K = cp.array(K_np, dtype=cp.float32)
        S = cp.random.rand(L, L, dtype=cp.float32) * 0.8
        
        avalanches = []

        # =========================================================
        # PRECOMPUTED FFT & BUFFERS (Pre-allocate to prevent latency)
        # =========================================================
        target_shape = 2 * L - 1
        fshape = (next_fast_len(target_shape), next_fast_len(target_shape))

        K_padded = cp.zeros(fshape, dtype=cp.float32)
        K_padded[:L, :L] = K
        K_fft = cp.fft.rfft2(K_padded)

        # Pre-allocate buffers once (Zero re-allocation inside the loop)
        tp = cp.zeros((L, L), dtype=cp.float32)
        tp_padded = cp.zeros(fshape, dtype=cp.float32)

        start_idx = (target_shape - L) // 2
        end_idx = start_idx + L
        # =========================================================

        for t in range(WARMUP + STEPS):
            # Injection
            xi = int(cp.random.randint(1, L-1))
            yi = int(cp.random.randint(1, L-1))
            S[xi, yi] += 0.85

            A = 0
            while True:
                # 🚀 EXTREME SPEED: Execute C++ CUDA Kernel for batch toppling
                get_topples(S, tp, S) 
                
                num_topple = int(tp.sum())
                if num_topple == 0: 
                    break

                A += num_topple

                # 🚀 Zero-allocation FFT Convolution
                tp_padded[:L, :L] = tp
                tp_fft = cp.fft.rfft2(tp_padded)
                spread_padded = cp.fft.irfft2(tp_fft * K_fft, s=fshape)
                
                S += spread_padded[start_idx:end_idx, start_idx:end_idx]

                # Absorbing boundaries
                S[0,:]=0; S[-1,:]=0; S[:,0]=0; S[:,-1]=0

            if t >= WARMUP and A > 2:
                avalanches.append(A)

            if t > 0 and t % 25000 == 0:
                status = "Warmup" if t < WARMUP else "Recording"
                elapsed = (time.time() - start_time) / 60
                print(f"    [{status}] Step {t}/{WARMUP+STEPS} | Elapsed: {elapsed:.2f} mins")

        # --- AUTO SAVE ---
        save_path = f"results/avalanches_alpha_{alpha}.csv"
        np.savetxt(save_path, avalanches, delimiter=",")
        print(f"    [Auto-Save] Data saved to {save_path}")

        tau = mle_tau(avalanches)
        total_time = (time.time() - start_time) / 60
        
        if tau is not None:
            print(f"  ↳ Result for alpha={alpha}: τ = {tau:.4f} (Events: {len(avalanches)}, Time: {total_time:.2f} mins)\n")
        else:
            print(f"  ↳ Result for alpha={alpha}: Not enough valid events. (Time: {total_time:.2f} mins)\n")
            
        sim_results.append((alpha, tau))

    return sim_results

# ══════════════════════════════════════════════════════════════════
# SECTION 398 — EMPIRICAL FITTING
# ══════════════════════════════════════════════════════════════════

def run_section_398_gpu_single(sim_results):
    print("\n" + "="*65)
    print("SECTION 398 — RESULTS SUMMARY")
    print("="*65)

    valid_data = [(a, t) for a, t in sim_results if t is not None]
    
    if len(valid_data) == 0:
        print("Error: No valid data points to display.")
        return
        
    if len(valid_data) < 3:
        print("Note: Skipping Curve Fitting (requires at least 3 data points).")
        print("Current Results:")
        for a, t in valid_data:
            print(f"  --> Alpha = {a} : Tau = {t:.4f}")
        return

# ══════════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_section_252_gpu_single(L=300)
    run_section_398_gpu_single(results)
    print("\nSIMULATION COMPLETE!")
