import numpy as np
import cupy as cp
from cupyx.scipy.fft import next_fast_len
import os
import time

# ==============================================================================
# CONFIGURATION & GPU OPTIMIZATION SETUP
# ==============================================================================
# Lock the Memory Pool to force CuPy to reserve RAM
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

# Process grid-wide toppling simultaneously at the C++ level.
# This works identically in 2D and 3D since it operates element-wise on the flat memory array.
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
# SECTION 252 — EXTREME GPU ACCELERATION (3D VERSION)
# ══════════════════════════════════════════════════════════════════

def run_section_252_gpu_single_3d(L=64):
    print("\n" + "="*65)
    print(f"🚀 SECTION 252 — EXTREME GPU 3D (L={L}) : ABELIAN MULTI-TOPPLING")
    print("="*65)
    print("Note: Powered by 3D Direct-View Padding & Pre-Allocated RNG Arrays.\n")

    alphas = [5.0] 
    sim_results = []

    # Simulation cycles
    WARMUP = 100000
    STEPS = 50000
    TOTAL_STEPS = WARMUP + STEPS

    for alpha in alphas:
        start_time = time.time()
        print(f"▶ Simulating alpha={alpha:.1f} on 3D Grid {L}x{L}x{L} ...")

        # 1. Create 3D Kernel Filter on CPU (Executed only once)
        z, y, x = np.ogrid[-L//2+1:L//2+1, -L//2+1:L//2+1, -L//2+1:L//2+1]
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-4
        cutoff = L / 4.0
        K_np = (r**-alpha) * np.exp(-r/cutoff)
        K_np[L//2-1, L//2-1, L//2-1] = 0.0
        K_np /= K_np.sum() 

        # 2. Transfer data to GPU & allocate in-place 3D memory
        K = cp.array(K_np, dtype=cp.float32)
        S = cp.random.rand(L, L, L, dtype=cp.float32) * 0.8
        
        # PRE-GENERATE RANDOM NUMBERS FOR 3D: Pre-allocate RAM to reduce CPU-GPU bottlenecks
        xi_arr = cp.random.randint(1, L-1, size=TOTAL_STEPS, dtype=cp.int32)
        yi_arr = cp.random.randint(1, L-1, size=TOTAL_STEPS, dtype=cp.int32)
        zi_arr = cp.random.randint(1, L-1, size=TOTAL_STEPS, dtype=cp.int32)

        avalanches = []

        # =========================================================
        # PRECOMPUTED 3D FFT & ZERO-COPY BUFFERS
        # =========================================================
        target_shape = 2 * L - 1
        nfl = next_fast_len(target_shape)
        fshape = (nfl, nfl, nfl) # 3D shape for FFT padding

        K_padded = cp.zeros(fshape, dtype=cp.float32)
        K_padded[:L, :L, :L] = K
        K_fft = cp.fft.rfftn(K_padded) # Use N-dimensional real FFT

        # 🚀 ZERO-COPY TRICK FOR 3D
        tp_padded = cp.zeros(fshape, dtype=cp.float32)
        tp_view = tp_padded[:L, :L, :L] 

        start_idx = (target_shape - L) // 2
        end_idx = start_idx + L
        # =========================================================

        for t in range(TOTAL_STEPS):
            # Fetch pre-generated random 3D coordinates
            S[xi_arr[t], yi_arr[t], zi_arr[t]] += 0.85

            A = 0
            while True:
                # 🚀 EXTREME SPEED: Run C++ CUDA Kernel writing directly to 3D tp_view
                get_topples(S, tp_view, S) 
                
                num_topple = int(tp_view.sum())
                if num_topple == 0: 
                    break

                A += num_topple

                # 🚀 3D FFT Convolution
                tp_fft = cp.fft.rfftn(tp_padded)
                spread_padded = cp.fft.irfftn(tp_fft * K_fft, s=fshape)
                
                # Add the spread results back to the 3D grid
                S += spread_padded[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]

                # Fast clearance of 3D boundaries (Absorbing boundary conditions in all 6 faces)
                S[0,:,:]=0; S[-1,:,:]=0 
                S[:,0,:]=0; S[:,-1,:]=0
                S[:,:,0]=0; S[:,:,-1]=0

            # Record Avalanche size
            if t >= WARMUP and A > 2:
                avalanches.append(A)

            # High-frequency progress reporting
            if t > 0 and t % 5000 == 0:
                status = "Warmup" if t < WARMUP else "Recording"
                elapsed = (time.time() - start_time) / 60
                print(f"    [{status}] Step {t}/{TOTAL_STEPS} | Elapsed: {elapsed:.2f} mins")

        # --- AUTO SAVE ---
        save_path = f"results/avalanches_3d_alpha_{alpha}.csv"
        np.savetxt(save_path, avalanches, delimiter=",")
        print(f"    [Auto-Save] Data saved to {save_path}")

        tau = mle_tau(avalanches)
        total_time = (time.time() - start_time) / 60
        
        if tau is not None:
            print(f"  ↳ Result for alpha={alpha}: τ = {tau:.4f} (Events: {len(avalanches)}, Time: {total_time:.2f} mins)\n")
        else:
            print(f"  ↳ Result for alpha={alpha}: Not enough valid events. (Time: {total_time:.2f} mins)\n")
            
        sim_results.append((alpha, tau))

    # Print VRAM usage report
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
    # Starting execution at grid size L=64 to respect 3D memory constraints.
    # (If your GPU has enough VRAM, you can try L=128)
    results = run_section_252_gpu_single_3d(L=64)
    run_section_398_gpu_single(results)
    print("\n✅ 3D SIMULATION COMPLETE!")
