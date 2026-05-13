import numpy as np
import cupy as cp
from cupyx.scipy.fft import next_fast_len
import os
import time

# ==============================================================================
# CONFIGURATION & GPU OPTIMIZATION SETUP
# ==============================================================================
# Lock the Memory Pool to force CuPy to reserve RAM, preventing the overhead 
# of repeatedly borrowing/returning memory to the OS.
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

# Process grid-wide toppling simultaneously at the C++ level
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
    print("Note: Powered by Direct-View Padding & Pre-Allocated RNG Arrays.\n")

    alphas = [5.0] 
    sim_results = []

    # Simulation cycles (For L=512, avalanches can be massive, so this might take time)
    WARMUP = 100000
    STEPS = 50000
    TOTAL_STEPS = WARMUP + STEPS

    for alpha in alphas:
        start_time = time.time()
        print(f"▶ Simulating alpha={alpha:.1f} on Grid {L}x{L} ...")

        # 1. Create Kernel Filter on CPU (Executed only once)
        y, x = np.ogrid[-L//2+1:L//2+1, -L//2+1:L//2+1]
        r = np.sqrt(x**2 + y**2) + 1e-4
        cutoff = L / 4.0
        K_np = (r**-alpha) * np.exp(-r/cutoff)
        K_np[L//2-1, L//2-1] = 0.0
        K_np /= K_np.sum() 

        # 2. Transfer data to GPU & allocate in-place memory
        K = cp.array(K_np, dtype=cp.float32)
        S = cp.random.rand(L, L, dtype=cp.float32) * 0.8
        
        # PRE-GENERATE RANDOM NUMBERS: Pre-allocate RAM to reduce CPU-GPU bottlenecks
        xi_arr = cp.random.randint(1, L-1, size=TOTAL_STEPS, dtype=cp.int32)
        yi_arr = cp.random.randint(1, L-1, size=TOTAL_STEPS, dtype=cp.int32)

        avalanches = []

        # =========================================================
        # PRECOMPUTED FFT & ZERO-COPY BUFFERS
        # =========================================================
        target_shape = 2 * L - 1
        fshape = (next_fast_len(target_shape), next_fast_len(target_shape))

        K_padded = cp.zeros(fshape, dtype=cp.float32)
        K_padded[:L, :L] = K
        K_fft = cp.fft.rfft2(K_padded)

        # 🚀 ZERO-COPY TRICK: Allocate tp_padded once, and use a memory view 
        # pointing to the inner region. This eliminates the need to copy 
        # tp into tp_padded inside every single sub-loop!
        tp_padded = cp.zeros(fshape, dtype=cp.float32)
        tp_view = tp_padded[:L, :L] 

        start_idx = (target_shape - L) // 2
        end_idx = start_idx + L
        # =========================================================

        for t in range(TOTAL_STEPS):
            # Fetch pre-generated random coordinates
            S[xi_arr[t], yi_arr[t]] += 0.85

            A = 0
            while True:
                # 🚀 EXTREME SPEED: Run C++ CUDA Kernel writing directly to tp_view
                # The area outside tp_view inside tp_padded safely remains 0 at all times
                get_topples(S, tp_view, S) 
                
                # Requires CPU-GPU synchronization to evaluate stopping condition
                # (casting to int is faster than float evaluation)
                num_topple = int(tp_view.sum())
                if num_topple == 0: 
                    break

                A += num_topple

                # 🚀 FFT Convolution (In-place operations prevent new VRAM allocations)
                tp_fft = cp.fft.rfft2(tp_padded)
                spread_padded = cp.fft.irfft2(tp_fft * K_fft, s=fshape)
                
                # Add the spread results back (Slicing is extremely fast on the GPU)
                S += spread_padded[start_idx:end_idx, start_idx:end_idx]

                # Fast clearance of boundaries (Absorbing boundary conditions)
                S[0,:]=0; S[-1,:]=0; S[:,0]=0; S[:,-1]=0

            # Record Avalanche size
            if t >= WARMUP and A > 2:
                avalanches.append(A)

            # High-frequency progress reporting to ensure the program is running actively
            if t > 0 and t % 5000 == 0:
                status = "Warmup" if t < WARMUP else "Recording"
                elapsed = (time.time() - start_time) / 60
                print(f"    [{status}] Step {t}/{TOTAL_STEPS} | Elapsed: {elapsed:.2f} mins")

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

    # Print VRAM usage report for the entire execution
    used_bytes = mempool.used_bytes()
    total_bytes = mempool.total_bytes()
    print(f"[VRAM Status] Used: {used_bytes/1e6:.2f} MB / Total Allocated: {total_bytes/1e6:.2f} MB")

    return sim_results

# ══════════════════════════════════════════════════════════════════
# SECTION 398 — EMPIRICAL FITTING
# ══════════════════════════════════════════════════════════════════

def run_section_398_gpu_single(sim_results):
    print("\n" + "="*65)
    print("📊 SECTION 398 — RESULTS SUMMARY")
    print("="*65)

    valid_data = [(a, t) for a, t in sim_results if t is not None]
    
    if len(valid_data) == 0:
        print("Error: No valid data points to display.")
        return
        
    print("Current Results:")
    for a, t in valid_data:
        print(f"  --> Alpha = {a} : Tau = {t:.4f}")

# ══════════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Start execution at grid size L=512
    results = run_section_252_gpu_single(L=512)
    run_section_398_gpu_single(results)
    print("\n✅ SIMULATION COMPLETE!")
