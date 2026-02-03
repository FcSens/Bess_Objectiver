"""
BESS Multi-Objective Optimization using NSGA-II

Optimizes Battery Energy Storage System (BESS) location and size using 
Non-dominated Sorting Genetic Algorithm II (NSGA-II) to balance four 
competing objectives:
  - F1: Net Present Value (maximize)
  - F2: Load Fluctuation (minimize)
  - F3: Voltage Violations (minimize)
  - F4: Global Warming Potential (minimize)

Usage:
    from bess_nsga2_optimizer import BESSOptimizer
    
    # Create optimizer
    optimizer = BESSOptimizer(
        load=load_data,
        solar=solar_data,
        wind=wind_data,
        candidate_buses=[1, 5, 10, 15, 20],
        capacity_range=(0.5, 10.0)
    )
    
    # Run optimization
    results = optimizer.run(
        n_generations=50,
        population_size=100
    )
    
    # Visualize results
    optimizer.plot_pareto_front()
    optimizer.save_results('bess_optimization_results.json')
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Sequence
import json
import time
from pathlib import Path
from collections import deque, OrderedDict

# Import modular configuration
from config import (
    OPTIMIZATION_CONFIG,
    THRESHOLD_CONFIG,
    RESULT_IDX_NPV,
    RESULT_IDX_FLUCTUATION,
    RESULT_IDX_VOLTAGE_TOTAL,
    RESULT_IDX_VOLTAGE_BUS,
    RESULT_IDX_GWP,
    RESULT_IDX_LCOS,
)

# Note: Module imports commented out - classes/functions defined inline below
# from optimization.callbacks import BESSProgressCallback, print_convergence_summary
# 
# from visualization.plots import (
#     plot_pareto_front_2d,
#     plot_pareto_front_3d,
#     plot_convergence_analysis,
# )

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    _PYMOO_AVAILABLE = True
except ImportError:
    _PYMOO_AVAILABLE = False
    logging.warning("pymoo not available. Install with: pip install pymoo")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available for plotting")

import base_functions as bf


# ============================================================================
# UTILITY CLASSES AND FUNCTIONS - Inline definitions for missing module imports
# ============================================================================

class BESSProgressCallback:
    """Callback for monitoring NSGA-II optimization progress.
    
    Tracks convergence metrics and provides periodic logging.
    """
    
    def __init__(self, log_interval: int = 5, convergence_window: int = 10):
        """Initialize callback.
        
        Args:
            log_interval: Print progress every N generations
            convergence_window: Window size for convergence detection
        """
        self.log_interval = log_interval
        self.convergence_window = convergence_window
        self.history = []
        self.generation_times = []
        self.start_time = time.time()
        
    def __call__(self, algorithm):
        """Called by pymoo after each generation."""
        gen = algorithm.n_gen
        n_evals = algorithm.evaluator.n_eval
        
        # Record metrics
        self.history.append({
            'generation': gen,
            'n_evals': n_evals,
            'n_solutions': len(algorithm.pop),
            'timestamp': time.time()
        })
        
        # Log progress
        if gen % self.log_interval == 0 or gen == 1:
            elapsed = time.time() - self.start_time
            logging.info(f"Generation {gen:3d} | Evaluations: {n_evals:5d} | "
                        f"Population: {len(algorithm.pop):3d} | Time: {elapsed:5.1f}s")
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get convergence analysis summary."""
        if not self.history:
            return {}
            
        return {
            'total_generations': len(self.history),
            'total_evaluations': self.history[-1]['n_evals'] if self.history else 0,
            'total_time': time.time() - self.start_time,
            'avg_time_per_generation': (time.time() - self.start_time) / len(self.history) if self.history else 0
        }


def print_convergence_summary(convergence: Dict[str, Any]):
    """Print convergence analysis summary.
    
    Args:
        convergence: Dictionary with convergence metrics
    """
    if not convergence:
        return
        
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    print(f"Total generations:     {convergence.get('total_generations', 'N/A')}")
    print(f"Total evaluations:     {convergence.get('total_evaluations', 'N/A')}")
    print(f"Total time:            {convergence.get('total_time', 0):.1f}s")
    print(f"Avg time/generation:   {convergence.get('avg_time_per_generation', 0):.2f}s")
    print("="*70 + "\n")


def plot_pareto_front_2d(
    pareto_f: np.ndarray,
    objective_names: List[str],
    title: str = "Pareto Front",
    filename: Optional[str] = None
):
    """Plot 2D Pareto front.
    
    Args:
        pareto_f: Objective values (N x M array)
        objective_names: Names of objectives
        title: Plot title
        filename: Optional filename to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available - skipping plot")
        return
    
    if pareto_f.shape[1] < 2:
        logging.warning("Need at least 2 objectives for 2D plot")
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for ax, (i, j) in zip(axes, pairs):
        if i < pareto_f.shape[1] and j < pareto_f.shape[1]:
            ax.scatter(pareto_f[:, i], pareto_f[:, j], alpha=0.6)
            ax.set_xlabel(objective_names[i] if i < len(objective_names) else f"F{i+1}")
            ax.set_ylabel(objective_names[j] if j < len(objective_names) else f"F{j+1}")
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logging.info(f"Saved 2D Pareto front to {filename}")
    else:
        plt.show()
    plt.close()


def plot_pareto_front_3d(
    pareto_f: np.ndarray,
    objective_names: List[str],
    title: str = "3D Pareto Front",
    filename: Optional[str] = None
):
    """Plot 3D Pareto front.
    
    Args:
        pareto_f: Objective values (N x M array, M >= 3)
        objective_names: Names of objectives
        title: Plot title
        filename: Optional filename to save plot
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logging.warning("matplotlib not available - skipping 3D plot")
        return
    
    if pareto_f.shape[1] < 3:
        logging.warning("Need at least 3 objectives for 3D plot")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pareto_f[:, 0], pareto_f[:, 1], pareto_f[:, 2], alpha=0.6)
    ax.set_xlabel(objective_names[0] if len(objective_names) > 0 else "F1")
    ax.set_ylabel(objective_names[1] if len(objective_names) > 1 else "F2")
    ax.set_zlabel(objective_names[2] if len(objective_names) > 2 else "F3")
    ax.set_title(title)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logging.info(f"Saved 3D Pareto front to {filename}")
    else:
        plt.show()
    plt.close()


def plot_convergence_analysis(
    convergence_data: Dict[str, Any],
    filename: Optional[str] = None
):
    """Plot convergence analysis.
    
    Args:
        convergence_data: Dictionary with convergence metrics
        filename: Optional filename to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available - skipping convergence plot")
        return
    
    # Simple convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If we have history data, plot it
    if 'history' in convergence_data:
        history = convergence_data['history']
        generations = [h['generation'] for h in history]
        ax.plot(generations, label='Progress')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Metric')
        ax.set_title('Convergence Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No convergence data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logging.info(f"Saved convergence plot to {filename}")
    else:
        plt.show()
    plt.close()


# BESSProgressCallback has been moved to optimization/callbacks.py
# Import it at the top of the file


class BESSOptimizationProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for BESS placement and sizing.
    
    Decision Variables:
        - capacity_mwh: Real (continuous) [capacity_min, capacity_max]
        - bus_index: Integer (discrete) [0, n_buses-1] (mapped to actual bus IDs)
    
    Objectives (all minimized in pymoo):
        - F1: -NPV (negative for minimization)
        - F2: Load Fluctuation
        - F3: Voltage Violations
        - F4: Global Warming Potential
    """
    
    def __init__(self,
             load: np.ndarray,
             solar: np.ndarray,
             wind: np.ndarray,
             candidate_buses: List[int],
             capacity_range: Tuple[float, float],
             grid_co2_intensity: Optional[np.ndarray] = None,
             network: Optional[Any] = None,
             use_pandapower: bool = True,
             use_cache: bool = True,
             n_representative_days: Optional[int] = None,
                 **kwargs):
        """
        Initialize BESS optimization problem.
        
        Args:
            load: Load time series (MW)
            solar: Solar generation time series (MW)
            wind: Wind generation time series (MW)
            candidate_buses: List of candidate bus IDs for BESS placement
            capacity_range: (min_capacity, max_capacity) in MWh
            grid_co2_intensity: Optional CO2 intensity time series (kgCO2/MWh)
            network: Pandapower network object
            use_pandapower: Whether to use pandapower simulations
            use_cache: Whether to use baseline caching
            n_representative_days: Number of representative days for fast optimization
        """
        # Validate inputs
        self._validate_inputs(load, solar, wind, candidate_buses, capacity_range)

        self.load_original = np.asarray(load)
        self.solar_original = np.asarray(solar) if solar is not None else None
        self.wind_original = np.asarray(wind) if wind is not None else None
        self.grid_co2 = grid_co2_intensity
        self.candidate_buses = list(candidate_buses)
        self.capacity_min, self.capacity_max = capacity_range
        self.network = network
        self.use_pandapower = use_pandapower
        self.n_representative_days = n_representative_days

        # Prepare optimization cache for speedup
        self.cache = None
        if use_cache:
            try:
                logging.info("Preparing optimization cache...")
                self.cache = bf.prepare_optimization_cache(
                    load=self.load_original,
                    solar=self.solar_original,
                    wind=self.wind_original,
                    network=network,
                    n_representative_days=n_representative_days
                )
                # Use cached/subset data
                self.load = self.cache['load']
                self.solar = self.cache['solar']
                self.wind = self.cache['wind']
                logging.info(f"Cache prepared: {self.cache['speedup_factor']:.1f}x expected speedup")

            except Exception as e:
                logging.warning(f"Cache preparation failed: {e}, using original data")
                self.load = self.load_original
                self.solar = self.solar_original
                self.wind = self.wind_original
        else:
            self.load = self.load_original
            self.solar = self.solar_original
            self.wind = self.wind_original

        # Statistics (use deque for memory efficiency)
        self.n_evaluations = 0
        self.evaluation_times = deque(maxlen=1000)  # Keep only last 1000 times
        
        # Track worst observed values for adaptive error penalties
        self.worst_observed = np.array([0.0, 0.0, 0.0, 0.0])  # [min_npv, max_fluct, max_volt, max_gwp]
        
        # Evaluation cache to avoid duplicate evaluations (OrderedDict for LRU behavior)
        self.evaluation_cache = OrderedDict()  # (capacity, bus) -> objectives
        self.cache_hits = 0
        self.cache_maxsize = OPTIMIZATION_CONFIG.cache_maxsize  # Use config value

        # Initialize pymoo ElementwiseProblem
        # 2 variables: [capacity (continuous), bus_idx (integer, will be rounded)]
        # 4 objectives: [F1_NPV, F2_Fluctuation, F3_Voltage, F4_GWP]
        super().__init__(
            n_var=2,
            n_obj=4,
            n_constr=0,
            xl=np.array([self.capacity_min, 0]),
            xu=np.array([self.capacity_max, len(self.candidate_buses) - 1]),
            **kwargs
        )
    
    def _validate_inputs(self, load, solar, wind, candidate_buses, capacity_range):
        """Validate input parameters."""
        # Check load data
        load_arr = np.asarray(load)
        if len(load_arr) == 0:
            raise ValueError("Load data cannot be empty")
        if np.any(load_arr < 0):
            raise ValueError("Load values must be non-negative")

        # Check solar/wind data consistency
        if solar is not None:
            solar_arr = np.asarray(solar)
            if len(solar_arr) != len(load_arr):
                raise ValueError(f"Solar data length ({len(solar_arr)}) must match load data length ({len(load_arr)})")
            if np.any(solar_arr < 0):
                raise ValueError("Solar generation values must be non-negative")
        
        if wind is not None:
            wind_arr = np.asarray(wind)
            if len(wind_arr) != len(load_arr):
                raise ValueError(f"Wind data length ({len(wind_arr)}) must match load data length ({len(load_arr)})")
            if np.any(wind_arr < 0):
                raise ValueError("Wind generation values must be non-negative")

        # Check candidate buses
        if len(candidate_buses) == 0:
            raise ValueError("Must provide at least one candidate bus")
        if len(set(candidate_buses)) != len(candidate_buses):
            raise ValueError("Candidate buses must be unique")

        # Check capacity range
        if capacity_range[0] >= capacity_range[1]:
            raise ValueError(f"Invalid capacity range: min ({capacity_range[0]}) must be less than max ({capacity_range[1]})")
        if capacity_range[0] <= 0:
            raise ValueError(f"Minimum capacity must be positive, got {capacity_range[0]}")
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate objectives for a single solution (elementwise).
        
        Args:
            x: Decision variable array [capacity, bus_idx]  
            out: Output dict to store objectives
        """
        capacity = float(x[0])
        
        # Protect against NaN/Inf in bus index (reliability fix)
        bus_val = x[1]
        if not np.isfinite(bus_val):
            logging.warning(f"Non-finite bus value detected: {bus_val}, defaulting to 0")
            bus_val = 0.0
        
        bus_idx = int(np.round(bus_val))  # Round to nearest integer
        bus_idx = np.clip(bus_idx, 0, len(self.candidate_buses) - 1)
        actual_bus = self.candidate_buses[bus_idx]
        
        # Check evaluation cache (improved precision for better hit rate)
        cache_key = (round(capacity, OPTIMIZATION_CONFIG.cache_precision), actual_bus)  # Use config precision
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            # Move to end for LRU behavior (most recently used)
            self.evaluation_cache.move_to_end(cache_key)
            objectives = self.evaluation_cache[cache_key]
            out["F"] = objectives
            
            # Update statistics without timing
            self.n_evaluations += 1
            if self.n_evaluations % 50 == 0:
                cache_rate = 100 * self.cache_hits / self.n_evaluations
                logging.debug("Cache hit rate: %.1f%% (%d/%d)", cache_rate, self.cache_hits, self.n_evaluations)
            return


        # Evaluate using base_functions
        t0 = time.time()
        try:
            # RELIABILITY FIX #4: Validate baseline cache network match
            baseline_to_use = None
            if self.cache and 'baseline' in self.cache:
                # Check if cache is from same network
                cache_network = self.cache.get('network_name', str(self.network))
                current_network = str(self.network)
                
                if cache_network == current_network:
                    baseline_to_use = self.cache['baseline']
                else:
                    logging.warning(
                        f"⚠ Baseline cache network mismatch!\n"
                        f"   Cache was prepared for: {cache_network}\n"
                        f"   Current network is: {current_network}\n"
                        f"   Re-computing baseline (slower but accurate)."
                    )
            
            result = bf.evaluate_objectives_simple(
                capacity_mwh=capacity,
                bus=actual_bus,
                load=self.load,
                solar=self.solar,
                wind=self.wind,
                grid_co2_intensity=self.grid_co2,
                use_pandapower=self.use_pandapower,
                network=self.network,
                return_profile=False,
                baseline_cache=baseline_to_use  # Use validated cache
            )
            
            # Validate result array length before indexing
            # Result can be list, tuple, or EvaluationResult object (with __getitem__ and __len__)
            try:
                result_len = len(result)
                if result_len < 7:
                    raise ValueError(f"Invalid result: expected 7 elements, got {result_len}")
            except TypeError:
                raise ValueError(f"Invalid result: expected sequence-like object with 7 elements, got {type(result).__name__}")
            
            # result = [capacity, F1_NPV, F2_fluctuation, F3_voltage_total, F3_voltage_bus, F4_GWP, LCOS]
            f1_npv = result[bf.RESULT_IDX_NPV]
            f2_fluct = result[bf.RESULT_IDX_FLUCTUATION]
            f3_volt = result[bf.RESULT_IDX_VOLTAGE_TOTAL]
            f4_gwp = result[bf.RESULT_IDX_GWP]
            
            # Convert to minimization (NSGA-II minimizes by default)
            objectives = np.array([
                -f1_npv,   # Maximize NPV -> minimize -NPV
                f2_fluct,  # Minimize fluctuation
                f3_volt,   # Minimize voltage violations
                f4_gwp     # Minimize GWP
            ])
            
            # Update worst observed values for adaptive penalties
            self.worst_observed[0] = min(self.worst_observed[0], -f1_npv)  # Min -NPV
            self.worst_observed[1] = max(self.worst_observed[1], f2_fluct)
            self.worst_observed[2] = max(self.worst_observed[2], f3_volt)
            self.worst_observed[3] = max(self.worst_observed[3], f4_gwp)
            
            # Store in cache with automatic LRU eviction (improved reliability)
            self.evaluation_cache[cache_key] = objectives
            # Enforce size limit - remove oldest if over capacity
            if len(self.evaluation_cache) > self.cache_maxsize:
                self.evaluation_cache.popitem(last=False)  # Remove oldest (FIFO)
                if self.n_evaluations % 1000 == 0:
                    logging.info("Cache evicted oldest entry (size: %d)", len(self.evaluation_cache))

        except Exception as e:
            logging.error(f"Evaluation failed for capacity={capacity}, bus={actual_bus}: {e}")
            # Use adaptive penalty: 10x worst observed (or large default if no observations)
            penalty = np.where(self.worst_observed > 0, 
                             self.worst_observed * 10, 
                             np.array([1e9, 1e9, 1e9, 1e9]))
            objectives = penalty
            
            # Don't cache failures
        
        self.evaluation_times.append(time.time() - t0)
        self.n_evaluations += 1

        # Reduce logging frequency for performance
        if self.n_evaluations % 100 == 0 and logging.getLogger().isEnabledFor(logging.INFO):
            avg_time = np.mean(self.evaluation_times) if len(self.evaluation_times) > 0 else 0.0
            logging.info("Evaluations: %d, Avg time: %.3fs", self.n_evaluations, avg_time)

        out["F"] = objectives


class BESSOptimizer:
    """
    High-level interface for BESS multi-objective optimization using NSGA-II.
    """
    
    def __init__(self,
                 load: Sequence[float],
                 solar: Optional[Sequence[float]] = None,
                 wind: Optional[Sequence[float]] = None,
                 grid_co2_intensity: Optional[Sequence[float]] = None,
                 candidate_buses: Optional[List[int]] = None,
                 capacity_range: Tuple[float, float] = (0.5, 10.0),
                 network: Optional[Any] = None,
                 use_pandapower: bool = True,
                 use_cache: bool = True,
                 n_representative_days: Optional[int] = 12):
        """
        Initialize BESS optimizer.
        
        Args:
            load: Load time series (MW)
            solar: Solar generation time series (MW)
            wind: Wind generation time series (MW)
            grid_co2_intensity: CO2 intensity time series (kgCO2/MWh)
            candidate_buses: List of candidate bus IDs (default: None = ALL buses in network)
            capacity_range: (min, max) capacity in MWh
            network: Pandapower network object or network name (default: case118)
            use_pandapower: Whether to use pandapower simulations
            use_cache: Whether to use optimization caching
            n_representative_days: Number of representative days (None = use all data)
        """
        if not _PYMOO_AVAILABLE:
            raise RuntimeError("pymoo is required for optimization. Install with: pip install pymoo")

        # Validate and clean input data
        self.load = self._validate_timeseries(load, "load", allow_none=False)
        self.solar = self._validate_timeseries(solar, "solar", allow_none=True)
        self.wind = self._validate_timeseries(wind, "wind", allow_none=True)
        self.grid_co2 = self._validate_timeseries(grid_co2_intensity, "grid_co2_intensity", allow_none=True)
        
        # Validate data consistency
        n_timesteps = len(self.load)
        if self.solar is not None and len(self.solar) != n_timesteps:
            raise ValueError(f"Solar data length ({len(self.solar)}) must match load length ({n_timesteps})")
        if self.wind is not None and len(self.wind) != n_timesteps:
            raise ValueError(f"Wind data length ({len(self.wind)}) must match load length ({n_timesteps})")
        if self.grid_co2 is not None and len(self.grid_co2) != n_timesteps:
            raise ValueError(f"CO2 intensity length ({len(self.grid_co2)}) must match load length ({n_timesteps})")
        
        # Auto-populate candidate buses from network if not specified
        if candidate_buses is None:
            if use_pandapower:
                try:
                    logging.info("No candidate buses specified - extracting ALL buses from network...")
                    self.candidate_buses = bf.get_all_buses_from_network(network)
                    logging.info(f"✓ Using all {len(self.candidate_buses)} buses as candidates")
                except Exception as e:
                    logging.warning(f"Failed to extract buses from network: {e}")
                    logging.warning("Falling back to default candidate buses")
                    self.candidate_buses = [1, 5, 10, 15, 20, 25, 30]
            else:
                # If not using pandapower, use default subset
                logging.warning("Not using pandapower - using default candidate buses")
                self.candidate_buses = [1, 5, 10, 15, 20, 25, 30]
        else:
            self.candidate_buses = candidate_buses
        
        if len(self.candidate_buses) == 0:
            raise ValueError("Must provide at least one candidate bus")
        if len(set(self.candidate_buses)) != len(self.candidate_buses):
            raise ValueError("Candidate buses must be unique")
        
        if capacity_range[0] >= capacity_range[1]:
            raise ValueError(f"Invalid capacity range: min ({capacity_range[0]}) >= max ({capacity_range[1]})")
        if capacity_range[0] <= 0:
            raise ValueError(f"Minimum capacity must be positive, got {capacity_range[0]}")
        
        self.capacity_range = capacity_range
        self.network = network
        self.use_pandapower = use_pandapower
        self.use_cache = use_cache
        self.n_representative_days = n_representative_days

        # RELIABILITY FIX #1: Validate candidate buses exist in network
        if self.use_pandapower and self.network and self.candidate_buses:
            try:
                _, _, validation_net = bf._init_pandapower_network(self.network)
                valid_buses = set(validation_net.bus.index.tolist())
                invalid_buses = [b for b in self.candidate_buses if b not in valid_buses]
                
                if invalid_buses:
                    raise ValueError(
                        f"❌ Invalid candidate buses: {invalid_buses}\n"
                        f"These buses do not exist in the network.\n"
                        f"Valid buses are: {sorted(valid_buses)}\n"
                        f"Please update candidate_buses parameter."
                    )
                
                logging.info(f"✓ Validated all {len(self.candidate_buses)} candidate buses exist in network")
            except ValueError:
                raise  # Re-raise ValueError for invalid buses
            except Exception as e:
                logging.warning(f"⚠ Could not validate buses against network: {e}")

        # Results storage
        self.result = None
        self.problem = None
        self.algorithm = None
        self.history = []
        self.callback = None  # Store callback for convergence metrics
    
    def _validate_timeseries(self, data: Optional[Sequence[float]], name: str, allow_none: bool = True) -> Optional[np.ndarray]:
        """
        Validate and clean timeseries data.
        
        Args:
            data: Input timeseries data
            name: Name for error messages
            allow_none: Whether None is acceptable
        
        Returns:
            Validated numpy array or None
        
        Raises:
            ValueError: If data is invalid
        """
        if data is None:
            if allow_none:
                return None
            else:
                raise ValueError(f"{name} data cannot be None")
        
        # Convert to numpy array
        try:
            arr = np.asarray(data, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{name} data could not be converted to numeric array: {e}")
        
        # Check dimensionality
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
        
        # Check not empty
        if len(arr) == 0:
            raise ValueError(f"{name} data cannot be empty")
        
        # Check for non-finite values
        if not np.all(np.isfinite(arr)):
            n_nan = np.sum(np.isnan(arr))
            n_inf = np.sum(np.isinf(arr))
            raise ValueError(f"{name} contains non-finite values: {n_nan} NaN, {n_inf} Inf")
        
        # Check for negative values
        if np.any(arr < 0):
            n_negative = np.sum(arr < 0)
            min_val = np.min(arr)
            raise ValueError(f"{name} contains {n_negative} negative values (min={min_val:.2f})")
        
        return arr
    
    
    def run(self,
            n_generations: int = 50,
            population_size: int = 100,
            crossover_prob: float = 0.9,
            mutation_prob: Optional[float] = None,
            seed: Optional[int] = 42,
            verbose: bool = True,
            n_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Run NSGA-II optimization.
        
        Args:
            n_generations: Number of generations
            population_size: Population size
            crossover_prob: Crossover probability (0-1)
            mutation_prob: Mutation probability (None = auto)
            seed: Random seed for reproducibility
            verbose: Whether to print progress
            n_workers: Number of parallel workers (None = auto, 1 = sequential)
        
        Returns:
            Dict containing optimization results
        """
        # Determine number of workers
        if n_workers is None:
            import multiprocessing
            n_workers = max(1, multiprocessing.cpu_count() - 1)
        
        if verbose:
            logging.info("="*70)
            logging.info("BESS MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)")
            logging.info("="*70)
            logging.info(f"Candidate buses: {self.candidate_buses}")
            logging.info(f"Capacity range: {self.capacity_range[0]:.2f} - {self.capacity_range[1]:.2f} MWh")
            logging.info(f"Population size: {population_size}")
            logging.info(f"Generations: {n_generations}")
            logging.info(f"Parallel workers: {n_workers}")
            logging.info(f"Data points: {len(self.load)}")
            if self.n_representative_days:
                logging.info(f"Using {self.n_representative_days} representative days for speed")

        # Create problem
        self.problem = BESSOptimizationProblem(
            load=self.load,
            solar=self.solar,
            wind=self.wind,
            grid_co2_intensity=self.grid_co2,
            candidate_buses=self.candidate_buses,
            capacity_range=self.capacity_range,
            network=self.network,
            use_pandapower=self.use_pandapower,
            use_cache=self.use_cache,
            n_representative_days=self.n_representative_days
        )

        # Configure NSGA-II algorithm
        if mutation_prob is None:
            mutation_prob = 1.0 / self.problem.n_var
        
        # Enable parallelization if workers > 1
        pool = None
        if n_workers > 1:
            try:
                from pymoo.core.problem import StarmapParallelization
                from multiprocessing.pool import ThreadPool
                
                pool = ThreadPool(n_workers)
                self.problem.elementwise_runner = StarmapParallelization(pool.starmap)
                logging.info(f"✓ Parallel evaluation enabled with {n_workers} workers")
            except ImportError:
                logging.warning("pymoo parallelization not available, running sequentially")
                pool = None
        
        self.algorithm = NSGA2(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=15),
            mutation=PM(prob=mutation_prob, eta=20),
            eliminate_duplicates=True
        )

        # Set up termination
        termination = get_termination("n_gen", n_generations)

        # Run optimization
        logging.info("\nStarting optimization...")
        start_time = time.time()
        
        # Add progress callback with convergence monitoring
        self.callback = BESSProgressCallback(
            log_interval=max(1, n_generations // 10),
            convergence_window=OPTIMIZATION_CONFIG.convergence_window
        ) if verbose else None
        
        try:
            # Build minimize arguments
            minimize_kwargs = {
                'problem': self.problem,
                'algorithm': self.algorithm,
                'termination': termination,
                'seed': seed,
                'verbose': False,  # Use our custom callback instead
                'save_history': True
            }
            
            # Only add callback if it exists
            if self.callback is not None:
                minimize_kwargs['callback'] = self.callback
            
            self.result = minimize(**minimize_kwargs)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        elapsed_time = time.time() - start_time

        # Process results
        results_dict = self._process_results(elapsed_time, verbose)
        
        # Add convergence summary if callback exists
        if self.callback and hasattr(self.callback, 'get_convergence_summary'):
            convergence_summary = self.callback.get_convergence_summary()
            results_dict['convergence'] = convergence_summary
            
            if verbose and convergence_summary:
                self._print_convergence_summary(convergence_summary)
        
        return results_dict
    
    def _process_results(self, elapsed_time: float, verbose: bool = True) -> Dict[str, Any]:
        """Process and summarize optimization results."""
        if self.result is None:
            raise ValueError("No optimization results available. Run optimization first.")

        # Extract Pareto front
        pareto_front = self.result.F
        pareto_solutions = self.result.X

        n_pareto = len(pareto_front)

        # Check for empty Pareto front
        if n_pareto == 0:
            logging.warning("No Pareto solutions found - optimization may have failed")
            return {
                'pareto_front': [],
                'pareto_solutions': {'capacity_mwh': [], 'bus': []},
                'best_solutions': {},
                'top_candidates': [],
                'statistics': {
                    'n_evaluations': self.problem.n_evaluations,
                    'n_pareto_solutions': 0,
                    'elapsed_time': elapsed_time,
                    'avg_evaluation_time': 0.0,
                    'total_evaluation_time': 0.0,
                    'hypervolume': 0.0
                },
                'configuration': {
                    'candidate_buses': self.candidate_buses,
                    'capacity_range': self.capacity_range,
                    'n_representative_days': self.n_representative_days,
                    'use_cache': self.use_cache
                }
            }
        
        # Extract solution components (handle array format with rounding for bus)
        pareto_capacities = pareto_solutions[:, 0]
        # Ensure bus indices are within valid range after rounding (with NaN protection)
        pareto_buses = []
        for x in pareto_solutions:
            bus_val = x[1]
            if not np.isfinite(bus_val):
                logging.warning(f"Non-finite bus value in Pareto solution: {bus_val}, using 0")
                bus_val = 0.0
            bus_idx = int(np.clip(np.round(bus_val), 0, len(self.candidate_buses) - 1))
            pareto_buses.append(self.candidate_buses[bus_idx])

        # Find best solutions for each objective
        best_indices = {
            'npv': np.argmin(pareto_front[:, 0]),      # Min -NPV = Max NPV
            'fluctuation': np.argmin(pareto_front[:, 1]),
            'voltage': np.argmin(pareto_front[:, 2]),
            'gwp': np.argmin(pareto_front[:, 3])
        }
        
        best_solutions = {}
        for obj_name, idx in best_indices.items():
            best_solutions[obj_name] = {
                'capacity_mwh': float(pareto_capacities[idx]),
                'bus': pareto_buses[idx],  # Already int from list comprehension
                'npv': float(-pareto_front[idx, 0]),  # Convert back to positive
                'fluctuation': float(pareto_front[idx, 1]),
                'voltage': float(pareto_front[idx, 2]),
                'gwp': float(pareto_front[idx, 3])
            }

        # Get top 10 candidates using different ranking methods
        top_candidates = self._get_top_candidates(pareto_front, pareto_capacities, pareto_buses, n_top=10)

        # Compute statistics
        stats = {
            'n_evaluations': self.problem.n_evaluations,
            'n_pareto_solutions': n_pareto,
            'elapsed_time': elapsed_time,
            'avg_evaluation_time': float(np.mean(self.problem.evaluation_times)) if len(self.problem.evaluation_times) > 0 else 0.0,
            'total_evaluation_time': float(sum(self.problem.evaluation_times)),
            'hypervolume': self._compute_hypervolume(pareto_front) if n_pareto > 0 else 0.0
        }
        
        results_dict = {
            'pareto_front': pareto_front.tolist(),
            'pareto_solutions': {
                'capacity_mwh': pareto_capacities.tolist(),
                'bus': pareto_buses
            },
            'best_solutions': best_solutions,
            'top_candidates': top_candidates,
            'statistics': stats,
            'configuration': {
                'candidate_buses': self.candidate_buses,
                'capacity_range': self.capacity_range,
                'n_representative_days': self.n_representative_days,
                'use_cache': self.use_cache
            }
        }

        if verbose:
            self._print_summary(results_dict)
            self._print_top_candidates(top_candidates)
            self._log_all_results(pareto_front, pareto_capacities, pareto_buses)
        
        return results_dict
    
    def _compute_hypervolume(self, pareto_front: np.ndarray) -> float:
        """Compute hypervolume indicator (quality metric for Pareto front)."""
        try:
            from pymoo.indicators.hv import HV
            # Reference point: slightly worse than worst values in Pareto front
            ref_point = np.max(pareto_front, axis=0) * 1.1
            hv = HV(ref_point=ref_point)
            return float(hv(pareto_front))
        except Exception as e:
            logging.warning(f"Hypervolume computation failed: {e}")
            return 0.0
    
    def _get_top_candidates(self, pareto_front: np.ndarray, 
                           pareto_capacities: np.ndarray, 
                           pareto_buses: List[int], 
                           n_top: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N candidate solutions using multiple ranking methods.
        
        Returns list of candidates with their rankings from different methods:
        - Utopian distance (balanced trade-off)
        - Knee point (steep trade-offs)
        - Weighted combinations
        """
        n_solutions = len(pareto_front)
        n_top = min(n_top, n_solutions)
        
        # Normalize objectives for fair comparison
        F_norm = self._normalize_objectives(pareto_front)
        
        # Method 1: Utopian point distance
        utopian = np.zeros(4)
        utopian_distances = np.sqrt(np.sum((F_norm - utopian)**2, axis=1))
        utopian_ranking = np.argsort(utopian_distances)
        
        # Method 2: Knee point (distance from origin in normalized space)
        knee_distances = np.sqrt(np.sum(F_norm**2, axis=1))
        knee_ranking = np.argsort(knee_distances)
        
        # Method 3: Financial emphasis (60% NPV, 40% others)
        financial_weights = np.array([0.6, 0.133, 0.133, 0.133])
        financial_scores = np.dot(F_norm, financial_weights)
        financial_ranking = np.argsort(financial_scores)
        
        # Method 4: Environmental emphasis (60% GWP, 40% others)
        environmental_weights = np.array([0.133, 0.133, 0.133, 0.6])
        environmental_scores = np.dot(F_norm, environmental_weights)
        environmental_ranking = np.argsort(environmental_scores)
        
        # Method 5: Technical emphasis (60% technical objectives)
        technical_weights = np.array([0.2, 0.3, 0.3, 0.2])
        technical_scores = np.dot(F_norm, technical_weights)
        technical_ranking = np.argsort(technical_scores)
        
        # Aggregate rankings
        candidates = []
        for idx in range(n_solutions):
            # Calculate average rank across methods
            ranks = [
                np.where(utopian_ranking == idx)[0][0],
                np.where(knee_ranking == idx)[0][0],
                np.where(financial_ranking == idx)[0][0],
                np.where(environmental_ranking == idx)[0][0],
                np.where(technical_ranking == idx)[0][0]
            ]
            avg_rank = np.mean(ranks)
            
            candidates.append({
                'rank': int(avg_rank),
                'solution_id': int(idx),
                'capacity_mwh': float(pareto_capacities[idx]),
                'bus': int(pareto_buses[idx]),
                'objectives': {
                    'npv': float(-pareto_front[idx, 0]),
                    'fluctuation': float(pareto_front[idx, 1]),
                    'voltage': float(pareto_front[idx, 2]),
                    'gwp': float(pareto_front[idx, 3])
                },
                'rankings': {
                    'utopian': int(np.where(utopian_ranking == idx)[0][0] + 1),
                    'knee': int(np.where(knee_ranking == idx)[0][0] + 1),
                    'financial': int(np.where(financial_ranking == idx)[0][0] + 1),
                    'environmental': int(np.where(environmental_ranking == idx)[0][0] + 1),
                    'technical': int(np.where(technical_ranking == idx)[0][0] + 1),
                    'average': float(avg_rank + 1)
                },
                'distances': {
                    'utopian': float(utopian_distances[idx]),
                    'knee': float(knee_distances[idx])
                }
            })
        
        # Sort by average rank and return top N
        candidates.sort(key=lambda x: x['rank'])
        return candidates[:n_top]
    
    def _print_top_candidates(self, top_candidates: List[Dict[str, Any]]):
        """Print top candidate solutions with detailed information."""
        print("\n" + "="*80)
        print("TOP 10 CANDIDATE SOLUTIONS (Ranked by Multiple Criteria)")
        print("="*80)
        print("\nThese solutions represent the best overall trade-offs across")
        print("financial, technical, and environmental objectives.\n")
        
        for i, candidate in enumerate(top_candidates, 1):
            print(f"\n{'='*80}")
            print(f"RANK #{i} - Solution ID: {candidate['solution_id']}")
            print(f"{'='*80}")
            print(f"Configuration:")
            print(f"  Capacity:     {candidate['capacity_mwh']:>8.2f} MWh")
            print(f"  Bus Location: {candidate['bus']:>8}")
            print(f"\nObjective Values:")
            print(f"  F1 (NPV):         {candidate['objectives']['npv']:>15,.2f} $ (financial)")
            print(f"  F2 (Fluctuation): {candidate['objectives']['fluctuation']:>15.4f} (technical)")
            print(f"  F3 (Voltage):     {candidate['objectives']['voltage']:>15.4f} pu-hr (technical)")
            print(f"  F4 (GWP):         {candidate['objectives']['gwp']:>15.2f} kgCO2e/MWh (environmental)")
            print(f"\nRankings by Method:")
            print(f"  Utopian Point:    #{candidate['rankings']['utopian']:>3} (balanced)")
            print(f"  Knee Point:       #{candidate['rankings']['knee']:>3} (steep trade-offs)")
            print(f"  Financial Focus:  #{candidate['rankings']['financial']:>3} (60% NPV)")
            print(f"  Environmental:    #{candidate['rankings']['environmental']:>3} (60% GWP)")
            print(f"  Technical Focus:  #{candidate['rankings']['technical']:>3} (60% tech)")
            print(f"  Average Rank:     #{candidate['rankings']['average']:>6.1f}")
        
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("• Top-ranked solutions have good performance across ALL objectives")
        print("• Check individual rankings to understand specific strengths")
        print("• Solutions ranked high in 'Financial' excel at NPV")
        print("• Solutions ranked high in 'Environmental' have low GWP")
        print("• Solutions ranked high in 'Technical' minimize fluctuation/voltage issues")
        print("• 'Utopian Point' ranking shows overall balanced performance")
        print("="*80 + "\n")
    
    def _log_all_results(self, pareto_front: np.ndarray, 
                        pareto_capacities: np.ndarray,
                        pareto_buses: List[int]):
        """Log all Pareto-efficient solutions to file and console."""
        n_solutions = len(pareto_front)
        
        # Create detailed log file
        log_filename = 'all_pareto_solutions.csv'
        
        logging.info(f"Logging all {n_solutions} Pareto-efficient solutions...")
        
        # Prepare data for CSV
        import pandas as pd
        
        data = {
            'Solution_ID': list(range(n_solutions)),
            'Capacity_MWh': pareto_capacities.tolist(),
            'Bus': pareto_buses,
            'F1_NPV_dollars': [-f[0] for f in pareto_front],
            'F2_Fluctuation': [f[1] for f in pareto_front],
            'F3_Voltage_pu_hr': [f[2] for f in pareto_front],
            'F4_GWP_kgCO2e_MWh': [f[3] for f in pareto_front]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(log_filename, index=False)
        
        logging.info(f"✓ All solutions saved to '{log_filename}'")
        
        # Print summary statistics
        print("\n" + "="*80)
        print(f"ALL PARETO SOLUTIONS SUMMARY ({n_solutions} solutions)")
        print("="*80)
        print("\nCapacity Distribution:")
        print(f"  Mean:   {df['Capacity_MWh'].mean():.2f} MWh")
        print(f"  Median: {df['Capacity_MWh'].median():.2f} MWh")
        print(f"  Range:  {df['Capacity_MWh'].min():.2f} - {df['Capacity_MWh'].max():.2f} MWh")
        
        print("\nBus Distribution:")
        bus_counts = df['Bus'].value_counts().sort_index()
        for bus, count in bus_counts.items():
            pct = (count / n_solutions) * 100
            print(f"  Bus {bus:2}: {count:3} solutions ({pct:5.1f}%)")
        
        print("\nObjective Ranges:")
        print(f"  NPV:         ${df['F1_NPV_dollars'].min():>12,.0f} to ${df['F1_NPV_dollars'].max():>12,.0f}")
        print(f"  Fluctuation: {df['F2_Fluctuation'].min():>12.4f} to {df['F2_Fluctuation'].max():>12.4f}")
        print(f"  Voltage:     {df['F3_Voltage_pu_hr'].min():>12.4f} to {df['F3_Voltage_pu_hr'].max():>12.4f} pu-hr")
        print(f"  GWP:         {df['F4_GWP_kgCO2e_MWh'].min():>12.2f} to {df['F4_GWP_kgCO2e_MWh'].max():>12.2f} kgCO2e/MWh")
        print("="*80 + "\n")
        
        return df
    
    def _print_convergence_summary(self, convergence: Dict[str, Any]):
        """Print detailed convergence analysis - delegates to optimization.callbacks."""
        print_convergence_summary(convergence)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print optimization summary."""
        stats = results['statistics']
        best = results['best_solutions']
        
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        print(f"Total evaluations: {stats['n_evaluations']}")
        print(f"Pareto solutions: {stats['n_pareto_solutions']}")
        print(f"Total time: {stats['elapsed_time']:.1f}s")
        print(f"Avg evaluation time: {stats['avg_evaluation_time']:.3f}s")
        print(f"Hypervolume: {stats['hypervolume']:.2e}")
        
        print("\n" + "="*70)
        print("BEST SOLUTIONS PER OBJECTIVE")
        print("="*70)
        
        for obj_name, sol in best.items():
            print(f"\nBest {obj_name.upper()}:")
            print(f"  Capacity: {sol['capacity_mwh']:.2f} MWh")
            print(f"  Bus: {sol['bus']}")
            print(f"  NPV: ${sol['npv']:,.2f}")
            print(f"  Fluctuation: {sol['fluctuation']:.4f}")
            print(f"  Voltage: {sol['voltage']:.4f} pu-hours")
            print(f"  GWP: {sol['gwp']:.2f} kgCO2e/MWh")
        
        print("\n" + "="*70)
    
    def select_best_solution(self, method: str = 'utopian', weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Select single best solution from Pareto front using decision-making method.
        
        Args:
            method: Selection method:
                - 'utopian': Distance to ideal point (all objectives = 0 after normalization)
                - 'weighted': Custom weighted sum (requires weights parameter)
                - 'knee': Knee point (maximum trade-off curvature)
            weights: Optional weights [w_npv, w_fluct, w_volt, w_gwp] for weighted method
        
        Returns:
            Dict containing selected solution details
        
        Example:
            >>> results = optimizer.run(...)
            >>> best = optimizer.select_best_solution(method='utopian')
            >>> print(f"Best solution: {best['capacity_mwh']} MWh at bus {best['bus']}")
        """
        if self.result is None:
            raise ValueError("No results available. Run optimization first.")
        
        pareto_front = self.result.F
        pareto_solutions = self.result.X
        
        if len(pareto_front) == 0:
            raise ValueError("No Pareto solutions found")
        
        if method == 'utopian':
            F_norm = self._normalize_objectives(pareto_front)
            utopian = np.zeros(4)
            distances = np.sqrt(np.sum((F_norm - utopian)**2, axis=1))
            best_idx = np.argmin(distances)
            logging.info(f"Utopian method: Selected solution {best_idx} with distance {distances[best_idx]:.4f}")
        
        elif method == 'knee':
            F_norm = self._normalize_objectives(pareto_front)
            distances = np.sqrt(np.sum(F_norm**2, axis=1))
            best_idx = np.argmin(distances)
            logging.info(f"Knee method: Selected solution {best_idx}")
        
        elif method == 'weighted':
            if weights is None:
                raise ValueError("weights parameter required for weighted method")
            if len(weights) != 4:
                raise ValueError(f"weights must have 4 elements, got {len(weights)}")
            weights_arr = np.array(weights) / np.sum(weights)
            F_norm = self._normalize_objectives(pareto_front)
            scores = np.dot(F_norm, weights_arr)
            best_idx = np.argmin(scores)
            logging.info(f"Weighted method: Selected solution {best_idx}")
        
        else:
            raise ValueError(f"Unknown method: '{method}'")
        
        capacity = float(pareto_solutions[best_idx, 0])
        
        # Protect against NaN/Inf in bus index
        bus_val = pareto_solutions[best_idx, 1]
        if not np.isfinite(bus_val):
            logging.warning(f"Non-finite bus value in best solution: {bus_val}, using 0")
            bus_val = 0.0
        
        bus_idx = int(np.clip(np.round(bus_val), 0, len(self.candidate_buses) - 1))
        bus = self.candidate_buses[bus_idx]
        
        return {
            'method': method,
            'pareto_index': int(best_idx),
            'capacity_mwh': capacity,
            'bus': bus,
            'objectives': {
                'npv': float(-pareto_front[best_idx, 0]),
                'fluctuation': float(pareto_front[best_idx, 1]),
                'voltage': float(pareto_front[best_idx, 2]),
                'gwp': float(pareto_front[best_idx, 3])
            }
        }
    
    def get_top_candidates(self, n_top: int = 10, 
                          ranking_method: str = 'aggregate') -> List[Dict[str, Any]]:
        """
        Get top N candidate solutions from Pareto front.
        
        Args:
            n_top: Number of top solutions to return (default: 10)
            ranking_method: Method to use for ranking:
                - 'aggregate': Average rank across all methods (recommended)
                - 'utopian': Utopian point distance (balanced)
                - 'knee': Knee point distance (steep trade-offs)
                - 'financial': Financial emphasis (60% NPV)
                - 'environmental': Environmental emphasis (60% GWP)
                - 'technical': Technical emphasis (60% technical objectives)
        
        Returns:
            List of top candidate solutions with detailed information
        
        Example:
            >>> results = optimizer.run(...)
            >>> top_10 = optimizer.get_top_candidates(n_top=10)
            >>> for candidate in top_10:
            ...     print(f"Rank {candidate['rank']}: {candidate['capacity_mwh']} MWh")
        """
        if self.result is None:
            raise ValueError("No results available. Run optimization first.")
        
        pareto_front = self.result.F
        pareto_solutions = self.result.X
        
        pareto_capacities = pareto_solutions[:, 0]
        
        # Protect against NaN/Inf in bus indices
        pareto_buses = []
        for x in pareto_solutions:
            bus_val = x[1]
            if not np.isfinite(bus_val):
                bus_val = 0.0
            bus_idx = int(np.clip(np.round(bus_val), 0, len(self.candidate_buses) - 1))
            pareto_buses.append(self.candidate_buses[bus_idx])
        
        # Get full candidate list with all rankings
        all_candidates = self._get_top_candidates(pareto_front, pareto_capacities, 
                                                  pareto_buses, n_top=len(pareto_front))
        
        # Sort by selected method
        if ranking_method == 'aggregate':
            all_candidates.sort(key=lambda x: x['rank'])
        elif ranking_method in ['utopian', 'knee', 'financial', 'environmental', 'technical']:
            all_candidates.sort(key=lambda x: x['rankings'][ranking_method])
        else:
            raise ValueError(f"Unknown ranking method: '{ranking_method}'. "
                           f"Use: 'aggregate', 'utopian', 'knee', 'financial', 'environmental', or 'technical'")
        
        return all_candidates[:n_top]
    
    def print_candidate_details(self, candidate: Dict[str, Any]):
        """
        Print detailed information about a specific candidate solution.
        
        Args:
            candidate: Candidate dict from get_top_candidates()
        
        Example:
            >>> top_10 = optimizer.get_top_candidates()
            >>> optimizer.print_candidate_details(top_10[0])  # Print best candidate
        """
        print("\n" + "="*80)
        print(f"CANDIDATE SOLUTION - ID: {candidate['solution_id']}")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  BESS Capacity:     {candidate['capacity_mwh']:.2f} MWh")
        print(f"  Installation Bus:  {candidate['bus']}")
        
        print(f"\nObjective Values:")
        print(f"  F1 - Net Present Value:    ${candidate['objectives']['npv']:>15,.2f}")
        print(f"       → Lifetime financial return")
        print(f"  F2 - Load Fluctuation:     {candidate['objectives']['fluctuation']:>15.4f}")
        print(f"       → Grid stability metric (lower = more stable)")
        print(f"  F3 - Voltage Violations:   {candidate['objectives']['voltage']:>15.4f} pu-hr")
        print(f"       → Power quality metric (lower = better quality)")
        print(f"  F4 - Global Warming:       {candidate['objectives']['gwp']:>15.2f} kgCO2e/MWh")
        print(f"       → Environmental impact (lower = greener)")
        
        print(f"\nRankings (out of {len(self.result.F)} Pareto solutions):")
        print(f"  Overall Average:           #{candidate['rankings']['average']:>6.1f}")
        print(f"  Utopian Point (balanced):  #{candidate['rankings']['utopian']:>6}")
        print(f"  Knee Point (trade-offs):   #{candidate['rankings']['knee']:>6}")
        print(f"  Financial Focus:           #{candidate['rankings']['financial']:>6}")
        print(f"  Environmental Focus:       #{candidate['rankings']['environmental']:>6}")
        print(f"  Technical Focus:           #{candidate['rankings']['technical']:>6}")
        
        print(f"\nQuality Metrics:")
        print(f"  Utopian Distance:          {candidate['distances']['utopian']:>15.4f}")
        print(f"  Knee Distance:             {candidate['distances']['knee']:>15.4f}")
        print("="*80 + "\n")
    
    def compare_candidates(self, candidate_ids: List[int]):
        """
        Compare multiple candidate solutions side-by-side.
        
        Args:
            candidate_ids: List of solution IDs to compare
        
        Example:
            >>> optimizer.compare_candidates([0, 5, 10, 15])  # Compare 4 solutions
        """
        if self.result is None:
            raise ValueError("No results available. Run optimization first.")
        
        pareto_front = self.result.F
        pareto_solutions = self.result.X
        
        print("\n" + "="*80)
        print("CANDIDATE COMPARISON")
        print("="*80)
        
        print(f"\n{'Metric':<25}", end='')
        for cid in candidate_ids:
            print(f"Solution {cid:>3}", end='  ')
        print()
        print("-"*80)
        
        # Capacity
        print(f"{'Capacity (MWh)':<25}", end='')
        for cid in candidate_ids:
            print(f"{pareto_solutions[cid, 0]:>12.2f}", end='  ')
        print()
        
        # Bus
        print(f"{'Bus Location':<25}", end='')
        for cid in candidate_ids:
            bus_val = pareto_solutions[cid, 1]
            if not np.isfinite(bus_val):
                bus_val = 0.0
            bus_idx = int(np.clip(np.round(bus_val), 0, len(self.candidate_buses) - 1))
            bus = self.candidate_buses[bus_idx]
            print(f"{bus:>12}", end='  ')
        print()
        
        print("-"*80)
        
        # NPV
        print(f"{'NPV ($)':<25}", end='')
        for cid in candidate_ids:
            print(f"{-pareto_front[cid, 0]:>12,.0f}", end='  ')
        print()
        
        # Fluctuation
        print(f"{'Fluctuation':<25}", end='')
        for cid in candidate_ids:
            print(f"{pareto_front[cid, 1]:>12.4f}", end='  ')
        print()
        
        # Voltage
        print(f"{'Voltage (pu-hr)':<25}", end='')
        for cid in candidate_ids:
            print(f"{pareto_front[cid, 2]:>12.4f}", end='  ')
        print()
        
        # GWP
        print(f"{'GWP (kgCO2e/MWh)':<25}", end='')
        for cid in candidate_ids:
            print(f"{pareto_front[cid, 3]:>12.2f}", end='  ')
        print()
        
        print("="*80 + "\n")
    
    def _normalize_objectives(self, F: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0, 1] range."""
        F_norm = np.zeros_like(F, dtype=float)
        for i in range(F.shape[1]):
            f_min, f_max = np.min(F[:, i]), np.max(F[:, i])
            F_norm[:, i] = (F[:, i] - f_min) / (f_max - f_min) if f_max > f_min else 0.0
        return F_norm
    
    
    def plot_pareto_front(self, 
                          pairs: Optional[List[Tuple[int, int]]] = None,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None):
        """
        Plot Pareto front for objective pairs.
        
        Delegates to visualization.plots.plot_pareto_front_2d()
        
        Args:
            pairs: List of (obj1, obj2) index pairs to plot (default: all pairs)
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        if self.result is None:
            raise ValueError("No results to plot. Run optimization first.")
        
        plot_pareto_front_2d(
            pareto_front=self.result.F,
            pairs=pairs,
            figsize=figsize,
            save_path=save_path
        )
    
    def plot_3d_pareto(self,
                      objectives: Tuple[int, int, int] = (0, 1, 2),
                      figsize: Tuple[int, int] = (12, 10),
                      save_path: Optional[str] = None):
        """
        Plot 3D Pareto front.
        
        Delegates to visualization.plots.plot_pareto_front_3d()
        
        Args:
            objectives: Tuple of 3 objective indices to plot
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        if self.result is None:
            raise ValueError("No results to plot. Run optimization first.")
        
        plot_pareto_front_3d(
            pareto_front=self.result.F,
            objectives=objectives,
            figsize=figsize,
            save_path=save_path
        )
    
    def plot_convergence(self, figsize: Tuple[int, int] = (15, 10),
                        save_path: Optional[str] = None):
        """
        Plot comprehensive convergence analysis with stability and accuracy metrics.
        
        Delegates to visualization.plots.plot_convergence_analysis()
        
        Shows:
        - Best objective values over time
        - Hypervolume convergence
        - Convergence rate
        - Stability score
        """
        if self.result is None:
            logging.error("No results to plot. Run optimization first.")
            return
        
        plot_convergence_analysis(
            callback_data=self.callback,
            result_history=self.result if hasattr(self.result, 'history') else None,
            figsize=figsize,
            save_path=save_path
        )
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert (dict, list, numpy array, numpy scalar, etc.)
        
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def save_results(self, filepath: str):
        """Save optimization results to JSON file."""
        if self.result is None:
            raise ValueError("No results to save. Run optimization first.")
        
        results = self._process_results(0, verbose=False)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logging.info(f"Results saved to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def export_pareto_solutions(self, filepath: str, format: str = 'csv'):
        """
        Export Pareto solutions to CSV or Excel.
        
        Args:
            filepath: Output file path
            format: 'csv' or 'excel'
        """
        if self.result is None:
            raise ValueError("No results to export. Run optimization first.")
        
        import pandas as pd
        
        pareto_front = self.result.F
        pareto_solutions = self.result.X
        
        # Extract solution components
        capacities = pareto_solutions[:, 0]
        buses = [self.candidate_buses[int(np.round(x[1]))] for x in pareto_solutions]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Solution_ID': range(len(pareto_front)),
            'Capacity_MWh': capacities,
            'Bus': buses,
            'NPV': -pareto_front[:, 0],  # Convert back to positive
            'Fluctuation': pareto_front[:, 1],
            'Voltage_Violations': pareto_front[:, 2],
            'GWP': pareto_front[:, 3]
        })
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Pareto solutions exported to {filepath}")


def main():
    """Example usage of BESS optimizer."""
    import argparse
    import warnings

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s %(levelname)s: %(message)s")
    
    # Suppress harmless pandapower dtype warnings
    warnings.filterwarnings('ignore', message='.*dtypes could not be corrected.*')
    logging.getLogger('pandapower').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="BESS Multi-Objective Optimization (NSGA-II)")
    parser.add_argument("--input-csv", type=str, 
                       default="Data_2024_33bus_scaled_fullyear.csv",
                       help="Input CSV file with load/solar/wind data")
    parser.add_argument("--generations", "-g", type=int, default=50,
                       help="Number of generations")
    parser.add_argument("--population", "-p", type=int, default=100,
                       help="Population size")
    parser.add_argument("--capacity-min", type=float, default=0.5,
                       help="Minimum capacity (MWh)")
    parser.add_argument("--capacity-max", type=float, default=10.0,
                       help="Maximum capacity (MWh)")
    parser.add_argument("--network", type=str, default="case33bw",
                       help="Pandapower network name (e.g., case33bw, case118, case_ieee30)")
    parser.add_argument("--buses", type=str, default="all",
                       help="Comma-separated list of candidate bus IDs, or 'all' for all buses in network (default: all)")
    parser.add_argument("--no-pandapower", action="store_true",
                       help="Use heuristic instead of pandapower")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable optimization caching")
    parser.add_argument("--representative-days", type=int, default=12,
                       help="Number of representative days (0 = use all data)")
    parser.add_argument("--output-dir", type=str, default="optimization_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--workers", "-w", type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 1)")

    args = parser.parse_args()

    # Parse candidate buses
    if args.buses.lower() == 'all':
        candidate_buses = None  # Will auto-populate from network
        logging.info("Using ALL buses from network as candidates")
    else:
        candidate_buses = [int(b.strip()) for b in args.buses.split(',')]
        logging.info(f"Using specified candidate buses: {candidate_buses}")

    # Load data
    if Path(args.input_csv).exists():
        logging.info(f"Loading data from {args.input_csv}")
        data = bf.load_timeseries_from_csv(args.input_csv)
        load = data['load']
        solar = data['solar']
        wind = data['wind']
    else:
        logging.warning(f"File {args.input_csv} not found, using sample data")
        # Generate 1 year of sample data (8760 hours)
        sample_data = bf._sample_timeseries(hours=8760)
        load = sample_data['load']
        solar = sample_data['solar']
        wind = sample_data['wind']

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create optimizer
    optimizer = BESSOptimizer(
        load=load,
        solar=solar,
        wind=wind,
        candidate_buses=candidate_buses,
        capacity_range=(args.capacity_min, args.capacity_max),
        network=args.network,  # Pass network name for bus extraction
        use_pandapower=not args.no_pandapower,
        use_cache=not args.no_cache,
        n_representative_days=args.representative_days if args.representative_days > 0 else None
    )

    # Run optimization
    results = optimizer.run(
        n_generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        n_workers=args.workers,
        verbose=True
    )

    # Save results
    optimizer.save_results(output_dir / "optimization_results.json")
    optimizer.export_pareto_solutions(output_dir / "pareto_solutions.csv", format='csv')

    # Generate plots
    try:
        optimizer.plot_pareto_front(save_path=str(output_dir / "pareto_front_2d.png"))
        optimizer.plot_3d_pareto(save_path=str(output_dir / "pareto_front_3d.png"))
        optimizer.plot_convergence(save_path=str(output_dir / "convergence.png"))
    except Exception as e:
        logging.warning(f"Plotting failed: {e}")

    logging.info(f"\nAll results saved to {output_dir}/")
    logging.info("Optimization complete!")

if __name__ == "__main__":
    main()
