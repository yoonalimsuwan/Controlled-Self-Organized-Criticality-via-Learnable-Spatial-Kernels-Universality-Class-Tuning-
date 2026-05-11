import numpy as np
import cupy as cp
from cupyx.scipy.fft import next_fast_len
import os
import time

# ==============================================================================
# CONFIGURATION & GPU OPTIMIZATION SETUP
# ==============================================================================
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

np.random.seed(42)
cp.random.seed(42)
os.makedirs("results", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def mle_tau(avalanches, xmin_pct=15):
    """Estimates the power-law exponent (tau) using Maximum Likelihood Estimation."""
    if len(avalanches) == 0: return None
    arr = np.array(avalanches, dtype=float)
    arr = arr[arr >= 1]
    if len(arr) < 50: return None
    xmin = max(5.0, np.percentile(arr, xmin_pct))
    arr = arr[arr >= xmin]
    if len(arr) < 20: return None
    return float(1.0 + len(arr) / np.sum(np.log(arr / xmin)))

# ══════════════════════════════════════════════════════════════════
# CUDA C++ KERNEL (EXTREME PERFORMANCE)
# ══════════════════════════════════════════════════════════════════

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
# SECTION 252 — EXTREME GPU ACCELERATION (L=512)
# ══════════════════════════════════════════════════════════════════

def run_section_252_gpu_single(L=512):
    print("\n" + "="*65)
    print(f"🚀 SECTION 252 — EXTREME GPU (L={L}) : ABELIAN MULTI-TOPPLING")
    print("="*65)
    print("Note: Powered by Direct-View Padding, Pre-Allocated RNG Arrays & Auto-Save.\n")

    alphas = [7.0]
    sim_results = []

    WARMUP = 100000
    STEPS = 50000
    TOTAL_STEPS = WARMUP + STEPS

    for alpha in alphas:
        start_time = time.time()
        print(f"▶ Simulating alpha={alpha:.1f} on Grid {L}x{L} ...")

        # 1. Create Kernel Filter on CPU
        y, x = np.ogrid[-L//2+1:L//2+1, -L//2+1:L//2+1]
        r = np.sqrt(x**2 + y**2) + 1e-4
        cutoff = L / 4.0
        K_np = (r**-alpha) * np.exp(-r/cutoff)
        K_np[L//2-1, L//2-1] = 0.0
        K_np /= K_np.sum()

        # 2. Transfer Kernel to GPU
        K = cp.array(K_np, dtype=cp.float32)

        # PRE-GENERATE RANDOM NUMBERS
        xi_arr = cp.random.randint(1, L-1, size=TOTAL_STEPS, dtype=cp.int32)
        yi_arr = cp.random.randint(1, L-1, size=TOTAL_STEPS, dtype=cp.int32)

        # ---------------------------------------------------------
        # AUTO-SAVE / RESUME LOGIC (CHECKPOINT)
        # ---------------------------------------------------------
        checkpoint_path = f"results/checkpoint_alpha_{alpha}.npz"
        
        if os.path.exists(checkpoint_path):
            print(f"   🔄 Found checkpoint! Loading state from {checkpoint_path}...")
            data = np.load(checkpoint_path)
            S = cp.array(data['S'], dtype=cp.float32)
            avalanches = data['avalanches'].tolist()
            start_step = int(data['step']) + 1
            print(f"   🔄 Resuming from step {start_step}/{TOTAL_STEPS}...")
        else:
            S = cp.random.rand(L, L, dtype=cp.float32) * 0.8
            avalanches = []
            start_step = 0
            print("   🆕 Starting a new simulation...")

        # PRECOMPUTED FFT & ZERO-COPY BUFFERS
        target_shape = 2 * L - 1
        fshape = (next_fast_len(target_shape), next_fast_len(target_shape))

        K_padded = cp.zeros(fshape, dtype=cp.float32)
        K_padded[:L, :L] = K
        K_fft = cp.fft.rfft2(K_padded)

        tp_padded = cp.zeros(fshape, dtype=cp.float32)
        tp_view = tp_padded[:L, :L]

        start_idx = (target_shape - L) // 2
        end_idx = start_idx + L

        # ---------------------------------------------------------
        # MAIN SIMULATION LOOP (Starting from start_step)
        # ---------------------------------------------------------
        for t in range(start_step, TOTAL_STEPS):
            S[xi_arr[t], yi_arr[t]] += 0.85

            A = 0
            while True:
                get_topples(S, tp_view, S)

                # CPU-GPU sync for stopping condition
                num_topple = int(tp_view.sum())
                if num_topple == 0:
                    break

                A += num_topple

                # FFT Convolution
                tp_fft = cp.fft.rfft2(tp_padded)
                spread_padded = cp.fft.irfft2(tp_fft * K_fft, s=fshape)
                S += spread_padded[start_idx:end_idx, start_idx:end_idx]

                # Absorbing boundary conditions
                S[0,:]=0; S[-1,:]=0; S[:,0]=0; S[:,-1]=0

            if t >= WARMUP and A > 2:
                avalanches.append(A)

            # Auto-save every 10,000 steps
            if t > 0 and t % 10000 == 0:
                status = "Warmup" if t < WARMUP else "Recording"
                elapsed = (time.time() - start_time) / 60
                print(f"    [{status}] Step {t}/{TOTAL_STEPS} | Elapsed (this run): {elapsed:.2f} mins")
                
                # Save checkpoint
                np.savez(checkpoint_path, 
                         step=t, 
                         avalanches=np.array(avalanches), 
                         S=cp.asnumpy(S))

        # --- SAVE FINAL RESULTS & CLEANUP CHECKPOINT ---
        save_path = f"results/avalanches_alpha_{alpha}.csv"
        np.savetxt(save_path, avalanches, delimiter=",")
        
        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        tau = mle_tau(avalanches)
        total_time = (time.time() - start_time) / 60
        sim_results.append((alpha, tau))

        if tau is not None:
            print(f"  ↳ Result for alpha={alpha}: τ = {tau:.4f} (Run Time: {total_time:.2f} mins)\n")

    return sim_results

def run_section_398_summary(sim_results):
    print("\n" + "="*65)
    print("📊 SECTION 398 — RESULTS SUMMARY")
    print("="*65)
    for a, t in sim_results:
        if t: print(f"  --> Alpha = {a} : Tau = {t:.4f}")

if __name__ == "__main__":
    results = run_section_252_gpu_single(L=512)
    run_section_398_summary(results)
    print("\n✅ SIMULATION FOR ALPHA 7.0 COMPLETE!")
