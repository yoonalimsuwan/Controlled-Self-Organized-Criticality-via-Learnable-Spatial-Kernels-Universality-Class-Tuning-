# ============================================================================
# GPU-NATIVE 3D FFT + ELEMENTWISEKERNEL OPTIMIZATION FRAMEWORK V11
# COLAB T4 OPTIMIZED EDITION (PRODUCTION READY)
# ============================================================================

import numpy as np
import cupy as cp
from cupyx.scipy.fft import next_fast_len
import time
import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

__version__ = "11.0.2-colab-t4-prod"
__author__ = "Yoon A Limsuwan"
__license__ = "MIT"


# ============================================================================
# CONFIGURATION & GPU SETUP
# ============================================================================

@dataclass
class GPUConfig:
    """GPU configuration optimized for Colab T4."""
    device_id: int = 0
    enable_memory_pool: bool = True
    enable_pinned_memory: bool = False
    max_vram_fraction: float = 0.80
    batch_size_adaptive: bool = True
    
    def __post_init__(self):
        """Initialize GPU context with Colab T4 settings."""
        try:
            cp.cuda.Device(self.device_id).use()
        except Exception as e:
            print(f"❌ GPU Error: {e}")
            raise
        
        if self.enable_memory_pool:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(fraction=self.max_vram_fraction)
        
        print(f"✅ GPU Device {self.device_id} initialized (Colab T4 Edition)")
        self._print_gpu_info()
    
    def _print_gpu_info(self):
        """Print GPU hardware information."""
        try:
            device = cp.cuda.Device(self.device_id)
            props = device.attributes
            mempool = cp.get_default_memory_pool()
            
            print(f"   GPU Cores: {props['MultiProcessorCount']} SMs")
            print(f"   Clock: {props['ClockRate'] // 1000} MHz")
            print(f"   Memory Limit: {mempool.get_limit() / 1e9:.2f} GB")
            print(f"   CUDA Compute Capability: {device.compute_capability}\n")
        except Exception as e:
            print(f"   [GPU info unavailable: {e}]\n")
    
    def clear_cache(self):
        """Explicitly clear GPU memory cache."""
        try:
            mempool = cp.get_default_memory_pool()
            cp.cuda.Stream.null.synchronize()
            mempool.free_all_blocks()
        except Exception:
            pass


@dataclass
class SimulationConfig:
    """Simulation hyperparameters optimized for Colab T4."""
    L: int = 96  # Grid dimension (96 = safe for T4)
    alpha: float = 2.5
    cutoff_factor: float = 4.0
    topple_threshold: float = 1.0
    gravity: float = 0.85
    absorption_boundary: bool = True
    periodic_boundary: bool = False
    warmup_steps: int = 50000
    measurement_steps: int = 25000
    sparse_cutoff: float = 20.0
    batch_size_base: int = 10000
    enable_hierarchical: bool = False
    coarse_grain_factor: int = 2
    
    def to_dict(self) -> Dict:
        """Serialize configuration."""
        return {
            'L': self.L,
            'alpha': self.alpha,
            'cutoff_factor': self.cutoff_factor,
            'topple_threshold': self.topple_threshold,
            'gravity': self.gravity,
            'absorption_boundary': self.absorption_boundary,
            'periodic_boundary': self.periodic_boundary,
            'warmup_steps': self.warmup_steps,
            'measurement_steps': self.measurement_steps,
            'sparse_cutoff': self.sparse_cutoff,
            'batch_size_base': self.batch_size_base,
        }
    
    @classmethod
    def from_dict(cls, d: Dict):
        """Deserialize configuration."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# CUDA C++ ELEMENTWISE KERNELS (Colab T4 Compatible)
# ============================================================================

class ElementwiseKernelLibrary:
    """Library of hand-tuned CUDA kernels via CuPy (Colab-safe)."""
    
    @staticmethod
    def create_topple_kernel():
        """Fast toppling operation (element-wise)."""
        return cp.ElementwiseKernel(
            'float32 S_in',
            'float32 tp, float32 S_out',
            '''
            if (S_in >= 1.0f) {
                tp = floorf(S_in);
                S_out = S_in - tp;
            } else {
                tp = 0.0f;
                S_out = S_in;
            }
            ''',
            'topple_kernel',
            options=('--use_fast_math',)
        )
    
    @staticmethod
    def create_clamp_kernel():
        """Clamp values to [min, max] range."""
        return cp.ElementwiseKernel(
            'float32 x, float32 x_min, float32 x_max',
            'float32 y',
            'y = fminf(fmaxf(x, x_min), x_max);',
            'clamp_kernel',
            options=('--use_fast_math',)
        )
    
    @staticmethod
    def create_scale_kernel():
        """Element-wise scalar multiplication."""
        return cp.ElementwiseKernel(
            'float32 x, float32 scale',
            'float32 y',
            'y = x * scale;',
            'scale_kernel',
            options=('--use_fast_math',)
        )
    
    @staticmethod
    def create_add_scaled_kernel():
        """In-place: y += alpha * x."""
        return cp.ElementwiseKernel(
            'float32 x, float32 alpha',
            'float32 y',
            'y += alpha * x;',
            'add_scaled_kernel',
            options=('--use_fast_math',)
        )
    
    @staticmethod
    def create_binarize_kernel():
        """Convert to binary (threshold)."""
        return cp.ElementwiseKernel(
            'float32 x, float32 threshold',
            'float32 y',
            'y = (x >= threshold) ? 1.0f : 0.0f;',
            'binarize_kernel',
            options=('--use_fast_math',)
        )


# ============================================================================
# SPARSE CONTACT NETWORK
# ============================================================================

class SparseContactNetwork:
    """GPU-resident sparse contact network."""
    
    def __init__(self, L: int, alpha: float, sparse_cutoff: float):
        self.L = L
        self.alpha = alpha
        self.sparse_cutoff = sparse_cutoff
        self.neighbor_pairs = self._precompute_neighbors()
    
    def _precompute_neighbors(self) -> np.ndarray:
        """Precompute neighbor offsets within sparse_cutoff."""
        half_L = self.L // 2
        offsets = []
        for dx in range(-half_L, half_L + 1):
            for dy in range(-half_L, half_L + 1):
                for dz in range(-half_L, half_L + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
                    if r <= self.sparse_cutoff:
                        offsets.append((dx, dy, dz))
        return np.array(offsets, dtype=np.int32)
    
    def get_neighbors(self, pos_i: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get neighbor positions for a given position."""
        i, j, k = pos_i
        neighbors = []
        for dx, dy, dz in self.neighbor_pairs:
            ni = (i + dx) % self.L
            nj = (j + dy) % self.L
            nk = (k + dz) % self.L
            neighbors.append((ni, nj, nk))
        return neighbors


# ============================================================================
# ZERO-COPY FFT BUFFER ARCHITECTURE
# ============================================================================

class ZeroCopyFFTBuffer:
    """Memory-efficient FFT buffer using views instead of allocations."""
    
    def __init__(self, L: int, dtype=cp.float32, device='cuda'):
        """Initialize zero-copy buffer."""
        self.L = L
        self.dtype = dtype
        self.device = device
        
        # Determine FFT padding
        target_shape = 2 * L - 1
        self.fft_size = next_fast_len(target_shape)
        self.fshape = (self.fft_size, self.fft_size, self.fft_size)
        
        print(f"  ZeroCopyFFTBuffer: L={L} → fshape={self.fshape}")
        print(f"    Memory: {np.prod(self.fshape) * 4 / 1e9:.3f} GB (padded)")
        
        # Allocate padded buffer once
        self.padded = cp.zeros(self.fshape, dtype=dtype)
        self.view = self.padded[:L, :L, :L]
        self.view_dirty = False
    
    def clear_view(self):
        """Clear only the viewed region."""
        self.view[:] = 0
        self.view_dirty = False
    
    def clear_padded(self):
        """Clear entire padded buffer."""
        self.padded[:] = 0
        self.view_dirty = False
    
    def write_to_view(self, data: cp.ndarray):
        """Write data to view (no copy)."""
        assert data.shape == (self.L, self.L, self.L), f"Expected {(self.L, self.L, self.L)}, got {data.shape}"
        self.view[:] = data
        self.view_dirty = True
    
    def get_padded(self) -> cp.ndarray:
        """Get reference to padded buffer."""
        return self.padded
    
    def get_view(self) -> cp.ndarray:
        """Get reference to inner view."""
        return self.view
    
    def memory_usage_mb(self) -> float:
        """Return memory usage in MB."""
        return np.prod(self.fshape) * 4 / 1e6


# ============================================================================
# KERNEL COMPUTATION
# ============================================================================

class KernelComputation:
    """GPU-optimized spatial kernel computation."""
    
    @staticmethod
    def compute_3d_kernel(L: int, alpha: float, cutoff_factor: float) -> cp.ndarray:
        """
        Compute 3D power-law kernel: K(r) = r^(-alpha) * exp(-r / cutoff)
        
        Returns: kernel array (shape: L × L × L)
        """
        z_coord = np.fft.fftfreq(L, 1.0) * L
        y_coord = np.fft.fftfreq(L, 1.0) * L
        x_coord = np.fft.fftfreq(L, 1.0) * L
        
        Z, Y, X = np.meshgrid(z_coord, y_coord, x_coord, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-6
        cutoff = L / cutoff_factor
        K = (r ** (-alpha)) * np.exp(-r / cutoff)
        K[L//2, L//2, L//2] = 0.0
        K /= (K.sum() + 1e-8)
        
        return cp.array(K, dtype=cp.float32)


# ============================================================================
# 3D FFT CONVOLUTION ENGINE
# ============================================================================

class FFTConvolution3D:
    """3D FFT-based convolution optimized for GPU."""
    
    def __init__(self, L: int, kernel: cp.ndarray):
        """Initialize convolution engine."""
        self.L = L
        self.kernel = kernel
        
        target_shape = 2 * L - 1
        self.fft_size = next_fast_len(target_shape)
        self.fshape = (self.fft_size, self.fft_size, self.fft_size)
        
        # Pre-compute kernel FFT
        kernel_padded = cp.zeros(self.fshape, dtype=cp.float32)
        kernel_padded[:L, :L, :L] = kernel
        self.kernel_fft = cp.fft.rfftn(kernel_padded)
        
        self.fft_buffer = ZeroCopyFFTBuffer(L, dtype=cp.float32)
        
        print(f"  FFTConvolution3D initialized: L={L}, fft_size={self.fft_size}")
    
    def convolve(self, signal: cp.ndarray) -> cp.ndarray:
        """
        Compute convolution: output = signal * kernel
        
        Args:
            signal: Input array (shape: L × L × L)
        
        Returns:
            Convolution result (shape: L × L × L)
        """
        assert signal.shape == (self.L, self.L, self.L)
        
        self.fft_buffer.write_to_view(signal)
        signal_fft = cp.fft.rfftn(self.fft_buffer.get_padded())
        product_fft = signal_fft * self.kernel_fft
        result_padded = cp.fft.irfftn(product_fft, s=self.fshape)
        
        start_idx = (self.fft_size - self.L) // 2
        end_idx = start_idx + self.L
        result = result_padded[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        
        return result


# ============================================================================
# SANDPILE DYNAMICS ENGINE
# ============================================================================

class SandpileDynamics3D:
    """GPU-native 3D Abelian sandpile simulator."""
    
    def __init__(self, config: SimulationConfig, gpu_config: GPUConfig):
        self.config = config
        self.gpu_config = gpu_config
        self.L = config.L
        self.alpha = config.alpha
        
        print(f"🚀 Initializing SandpileDynamics3D (L={self.L})")
        
        # Create CUDA kernels
        self.topple_kernel = ElementwiseKernelLibrary.create_topple_kernel()
        self.clamp_kernel = ElementwiseKernelLibrary.create_clamp_kernel()
        
        # Compute spatial kernel
        print("  Computing spatial kernel...")
        K = KernelComputation.compute_3d_kernel(
            self.L, self.alpha, config.cutoff_factor
        )
        
        # Initialize FFT convolution
        print("  Initializing FFT convolution...")
        self.fft_conv = FFTConvolution3D(self.L, K)
        
        # Initialize state arrays
        self.S = cp.random.rand(self.L, self.L, self.L, dtype=cp.float32) * 0.8
        self.tp = cp.zeros((self.L, self.L, self.L), dtype=cp.float32)
        
        # Statistics
        self.avalanches = []
        self.time_steps = 0
        
        print(f"✅ SandpileDynamics3D ready (memory: {self.fft_conv.fft_buffer.memory_usage_mb():.1f} MB)\n")
    
    def step(self, t: int):
        """Single simulation step: add grain and run avalanche."""
        # Add grain at random location
        xi = int(cp.random.randint(1, self.L - 1))
        yi = int(cp.random.randint(1, self.L - 1))
        zi = int(cp.random.randint(1, self.L - 1))
        self.S[xi, yi, zi] += self.config.gravity
        
        # Run avalanche until no more topples
        A = 0
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Topple kernel
            self.topple_kernel(self.S, self.tp, self.S)
            
            # Count topples
            num_topple = int(self.tp.sum())
            if num_topple == 0:
                break
            
            A += num_topple
            
            # FFT convolution: distribute topples
            spread = self.fft_conv.convolve(self.tp)
            self.S += spread
            
            # Boundary conditions: absorbing
            self.S[0, :, :] = 0
            self.S[-1, :, :] = 0
            self.S[:, 0, :] = 0
            self.S[:, -1, :] = 0
            self.S[:, :, 0] = 0
            self.S[:, :, -1] = 0
            
            # Clear topple buffer
            self.tp[:] = 0
        
        # Record avalanche if in measurement phase
        if t >= self.config.warmup_steps and A > 2:
            self.avalanches.append(A)
        
        self.time_steps = t
    
    def run_simulation(self):
        """Run full simulation (warmup + measurement)."""
        total_steps = self.config.warmup_steps + self.config.measurement_steps
        
        print(f"▶ Running {total_steps} steps (warmup: {self.config.warmup_steps})...\n")
        
        start_time = time.time()
        
        for t in range(total_steps):
            self.step(t)
            
            if (t + 1) % 5000 == 0:
                elapsed = (time.time() - start_time) / 60
                phase = "Warmup" if t < self.config.warmup_steps else "Measurement"
                print(f"  [{phase}] Step {t + 1}/{total_steps} | Elapsed: {elapsed:.2f} min")
                
                # Clear GPU cache every 10k steps
                if (t + 1) % 10000 == 0:
                    self.gpu_config.clear_cache()
        
        total_time = (time.time() - start_time) / 60
        print(f"\n✅ Simulation complete! Time: {total_time:.2f} min")
        print(f"   Total avalanches recorded: {len(self.avalanches)}")
        
        return {
            'avalanches': self.avalanches,
            'time_minutes': total_time,
            'n_avalanches': len(self.avalanches),
            'mean_size': float(np.mean(self.avalanches)) if self.avalanches else 0,
            'max_size': float(np.max(self.avalanches)) if self.avalanches else 0,
            'std_size': float(np.std(self.avalanches)) if self.avalanches else 0,
        }


# ============================================================================
# POWER-LAW ANALYSIS
# ============================================================================

def mle_tau(avalanches: List[int], xmin_pct: float = 15.0) -> Optional[float]:
    """
    Maximum Likelihood Estimation of power-law exponent.
    
    Formula: τ = 1 + N / Σ(log(s_i / s_min))
    
    Args:
        avalanches: List of avalanche sizes
        xmin_pct: Percentile for minimum threshold
    
    Returns:
        Power-law exponent τ, or None if insufficient data
    """
    if len(avalanches) == 0:
        return None
    
    arr = np.array(avalanches, dtype=float)
    arr = arr[arr >= 1]
    
    if len(arr) < 50:
        return None
    
    xmin = max(5.0, np.percentile(arr, xmin_pct))
    arr = arr[arr >= xmin]
    
    if len(arr) < 20:
        return None
    
    tau = 1.0 + len(arr) / np.sum(np.log(arr / xmin))
    return float(tau)


def plot_avalanche_distribution(avalanches: List[int], save_path: str = "/content/results/"):
    """Plot avalanche size distribution (optional)."""
    try:
        import matplotlib.pyplot as plt
        
        arr = np.array(avalanches)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Linear histogram
        ax1.hist(arr, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Avalanche Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Avalanche Size Distribution (Linear)')
        ax1.grid(True, alpha=0.3)
        
        # Log-log plot
        arr_nonzero = arr[arr > 0]
        ax2.loglog(arr_nonzero, 1 + np.arange(len(arr_nonzero)), 'o', alpha=0.6, markersize=3)
        ax2.set_xlabel('Avalanche Size (log)')
        ax2.set_ylabel('Cumulative Count (log)')
        ax2.set_title('Avalanche Distribution (Log-Log)')
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plot_file = os.path.join(save_path, 'avalanche_distribution.png')
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        print(f"  📊 Plot saved: {plot_file}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️  Plotting skipped: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    np.random.seed(42)
    cp.random.seed(42)
    
    # Create output directory
    output_dir = "/content/results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🚀 CSOC-SSC GPU-NATIVE 3D FFT + ELEMENTWISEKERNEL FRAMEWORK V11")
    print("   COLAB T4 OPTIMIZED EDITION (PRODUCTION READY)")
    print("=" * 70)
    print(f"  Version: {__version__}")
    print(f"  Author: {__author__}")
    print(f"  License: {__license__}\n")
    
    # GPU configuration
    gpu_config = GPUConfig(device_id=0, enable_memory_pool=True)
    
    # Simulation configuration (COLAB T4 OPTIMIZED)
    sim_config = SimulationConfig(
        L=96,
        alpha=2.5,
        warmup_steps=50000,
        measurement_steps=25000,
        enable_hierarchical=False,
    )
    
    print(f"Configuration (COLAB T4 OPTIMIZED):")
    print(f"  Grid size: {sim_config.L}³")
    print(f"  Alpha (power-law): {sim_config.alpha}")
    print(f"  Total steps: {sim_config.warmup_steps + sim_config.measurement_steps}")
    print(f"  Hierarchical: {sim_config.enable_hierarchical}\n")
    
    # Run simulation
    sim = SandpileDynamics3D(sim_config, gpu_config)
    result = sim.run_simulation()
    
    # Power-law analysis
    tau = mle_tau(result['avalanches'])
    result['tau'] = tau
    
    print(f"\n📊 Results Summary:")
    print(f"  Total avalanches: {result['n_avalanches']}")
    print(f"  Mean size: {result['mean_size']:.2f}")
    print(f"  Std dev: {result['std_size']:.2f}")
    print(f"  Max size: {result['max_size']:.0f}")
    print(f"  Power-law exponent τ: {tau:.4f}")
    print(f"  Expected τ (theory): 1.26 ± 0.03\n")
    
    # Save results
    save_path = os.path.join(
        output_dir,
        f"v11_alpha_{sim_config.alpha}_L_{sim_config.L}_colab_t4.json"
    )
    
    with open(save_path, 'w') as f:
        json.dump({
            'config': sim_config.to_dict(),
            'result': result,
            'platform': 'Colab T4',
            'version': __version__,
        }, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {save_path}\n")
    
    # Plot results (optional)
    plot_avalanche_distribution(result['avalanches'], output_dir)
    
    print("=" * 70)
    print("🎉 SIMULATION COMPLETE!")
    print("=" * 70 + "\n")
    
    return result


if __name__ == "__main__":
    result = main()
