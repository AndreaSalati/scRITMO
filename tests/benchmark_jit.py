#!/usr/bin/env python3
"""
Benchmark script for JIT optimizations in scRITMO.

This script benchmarks the training performance of ContextModel
with and without JIT optimizations.

Usage:
    python tests/benchmark_jit.py [--device {cuda,cpu}] [--n_cells N] [--epochs E]

Example:
    python tests/benchmark_jit.py --device cuda --n_cells 10000 --epochs 100
"""

import time
import argparse
import numpy as np
import torch
import anndata
import pandas as pd
from scipy.stats import vonmises
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import scritmo as sr
from scritmo.ml.warmup import warmup_and_train
from scritmo import Beta


def simulate_synthetic_data(
    n_cells=10000,
    n_genes=50,
    n_harmonics=1,
    concentration=5.0,  # von Mises concentration
    seed=42,
):
    """
    Simulate synthetic single-cell rhythmic data.
    
    Returns:
        adata: AnnData object with simulated data
        true_phases: True cell phases
        params_g: Gene parameters (Beta DataFrame)
    """
    rng = np.random.RandomState(seed)
    
    # Sample true phases from von Mises distribution
    true_phases = vonmises.rvs(concentration, size=n_cells, random_state=rng)
    true_phases = true_phases % (2 * np.pi)
    
    # Create gene parameters
    genes = [f"gene_{i:03d}" for i in range(n_genes)]
    
    # Base expression levels
    a_0 = rng.uniform(1.0, 3.0, n_genes)
    
    # Amplitudes
    amps = rng.uniform(0.2, 1.0, n_genes)
    
    # Phases (acrophase)
    phases = rng.uniform(0, 2 * np.pi, n_genes)
    
    # Create Beta DataFrame
    params_data = {
        'a_0': a_0,
        'amp': amps,
        'phase': phases,
    }
    
    # Add harmonic coefficients
    for h in range(1, n_harmonics + 1):
        params_data[f'a_{h}'] = amps * np.cos(h * phases)
        params_data[f'b_{h}'] = amps * np.sin(h * phases)
    
    params_g = pd.DataFrame(params_data, index=genes)
    params_g = Beta(params_g)
    
    # Simulate counts
    # Simple model: expression varies by phase
    X = np.zeros((n_cells, n_genes))
    for i, gene in enumerate(genes):
        mean_expr = np.exp(a_0[i] + amps[i] * np.cos(true_phases - phases[i]))
        # Add noise
        counts = rng.poisson(mean_expr * 100)  # Scale factor for counts
        X[:, i] = counts
    
    # Create AnnData
    obs = pd.DataFrame({
        'cell_id': [f'cell_{i:05d}' for i in range(n_cells)],
        'true_phase': true_phases,
        'context': 'cell_type_1',  # Single context for simplicity
    })
    var = pd.DataFrame(index=genes)
    
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers['spliced'] = X.astype(np.float32)
    
    # Add total counts
    adata.obs['total_counts'] = X.sum(axis=1)
    
    return adata, true_phases, params_g


def benchmark_training(
    device='cuda',
    n_cells=10000,
    n_genes=50,
    batch_size=128,
    n_epochs=100,
    n_harmonics=1,
    seed=42,
):
    """
    Benchmark training performance.
    
    Returns:
        dict with timing results
    """
    print(f"\n{'='*60}")
    print(f"Benchmark Configuration:")
    print(f"  Device: {device}")
    print(f"  Cells: {n_cells}")
    print(f"  Genes: {n_genes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"{'='*60}\n")
    
    # Simulate data
    print("Simulating synthetic data...")
    adata, true_phases, params_g = simulate_synthetic_data(
        n_cells=n_cells,
        n_genes=n_genes,
        n_harmonics=n_harmonics,
        seed=seed,
    )
    print(f"Data shape: {adata.shape}")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Context (all same cell type)
    context = np.array(['cell_type_1'] * n_cells)
    
    # Warmup and train
    print(f"\nStarting training on {device}...")
    
    # Time the training
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.perf_counter()
    
    cmodel, losses, mad_epochs = warmup_and_train(
        adata=adata,
        params_g=params_g,
        context=context,
        context_mode='none',
        fix_phase=False,
        noise_model='nb',
        fix_disp_val='gene',
        log_amp_fn='logit',
        counts=None,
        pretrain_epochs=0,
        pretrain_batch_size=batch_size,
        n_epochs=n_epochs,
        layer='spliced',
        unspliced_layer=None,
        n_theta=24,
        batch_size=batch_size,
        learning_rate=0.001,
        true_phase=true_phases,
        init_mean=True,
        kill_amps=False,
        device=device,
        return_data=False,
        n_theta_post=24,
        weights_g=None,
        fixed_cell_phases=None,
    )
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    time_per_epoch = total_time / n_epochs
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Time per epoch: {time_per_epoch:.3f} seconds")
    print(f"  Final loss: {losses[-1]:.4f}")
    if mad_epochs is not None:
        print(f"  Final MAD: {mad_epochs[-1]:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'device': device,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'total_time': total_time,
        'time_per_epoch': time_per_epoch,
        'final_loss': float(losses[-1]),
        'losses': losses,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark JIT optimizations in scRITMO'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda if available)'
    )
    parser.add_argument(
        '--n_cells',
        type=int,
        default=10000,
        help='Number of cells to simulate (default: 10000)'
    )
    parser.add_argument(
        '--n_genes',
        type=int,
        default=50,
        help='Number of genes to simulate (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training (default: 128)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results (optional)'
    )
    
    args = parser.parse_args()
    
    # Print JIT status
    print("\n" + "="*60)
    print("JIT Optimization Status:")
    print("="*60)
    
    # Check if JIT functions exist
    try:
        from scritmo.ml.context_model import compute_nb_params
        print("  ✓ compute_nb_params: JIT-compiled")
    except ImportError:
        print("  ✗ compute_nb_params: Not found")
    
    try:
        from scritmo.ml.marginalization import vectorized_simpson_jit
        print("  ✓ vectorized_simpson_jit: JIT-compiled")
    except ImportError:
        print("  ✗ vectorized_simpson_jit: Not found")
    
    try:
        from scritmo.ml.misc.power_spherical.power_spherical import log_von_mises_jit
        print("  ✓ log_von_mises_jit: JIT-compiled")
    except ImportError:
        print("  ✗ log_von_mises_jit: Not found")
    
    print("="*60 + "\n")
    
    # Run benchmark
    results = benchmark_training(
        device=args.device,
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        seed=args.seed,
    )
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_json = {k: v.tolist() if hasattr(v, 'tolist') else v 
                          for k, v in results.items()}
            json.dump(results_json, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results


if __name__ == '__main__':
    main()
