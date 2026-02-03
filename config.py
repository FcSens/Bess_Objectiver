"""
Configuration and constants for BESS Multi-Objective Optimization.

This module centralizes all configuration parameters, constants, and default values
used throughout the BESS optimization system.
"""

from dataclasses import dataclass
from typing import Dict, Any


# ======================== BESS PARAMETERS ========================

@dataclass(frozen=True)
class BessConfig:
    """BESS operational parameters and constraints."""
    c_rate: float = 0.25          # C-rate for charging/discharging
    rte: float = 0.92             # Round-trip efficiency
    soc_min: float = 0.15         # Minimum state of charge (fraction)
    soc_max: float = 0.85         # Maximum state of charge (fraction)
    initial_soc: float = 0.50     # Initial state of charge (fraction)


# ======================== FINANCIAL PARAMETERS ========================

@dataclass(frozen=True)
class FinancialConfig:
    """Financial parameters for NPV and LCOS calculations."""
    capex_per_mwh: float = 285000.0    # Capital cost per MWh
    om_per_mwh: float = 10.0           # Operations & maintenance cost per MWh per year
    discount_rate: float = 0.07        # Discount rate (7%)
    project_life: int = 15             # Project lifetime in years
    lifetime_factor: float = 0.90      # Degradation factor


@dataclass(frozen=True)
class SourceConfig:
    """Grid and energy source parameters."""
    cost_conventional: float = 250.0   # Conventional energy cost ($/MWh)
    te_e: float = 0.0                  # Temperature coefficient


# ======================== ENVIRONMENTAL PARAMETERS ========================

@dataclass(frozen=True)
class EnvironmentalConfig:
    """Environmental impact parameters for GWP calculations."""
    emb_gwp_per_mwh_cap: float = 85000.0      # Embodied GWP per MWh capacity (kgCO2e)
    recycling_credit_factor: float = 0.15     # Recycling credit factor
    eol_transport_per_mwh: float = 100.0      # End-of-life transport emissions (kgCO2e/MWh)


# ======================== NETWORK PARAMETERS ========================

@dataclass(frozen=True)
class VoltageConfig:
    """Voltage limits for power flow analysis."""
    min_pu: float = 0.95    # Minimum voltage (per-unit)
    max_pu: float = 1.05    # Maximum voltage (per-unit)


@dataclass(frozen=True)
class NetworkConfig:
    """Power network configuration."""
    default_network: str = "case33bw"
    default_bus_index: int = 1
    default_timestep_hours: float = 1.0


# ======================== OPTIMIZATION PARAMETERS ========================

@dataclass(frozen=True)
class OptimizationConfig:
    """Optimization algorithm parameters."""
    # Cache settings
    cache_maxsize: int = 10000
    
    # Progress reporting
    log_interval: int = 5
    
    # Convergence monitoring
    convergence_window: int = 10
    convergence_threshold: float = 0.001
    stability_threshold: float = 0.01
    
    # Power flow settings
    max_pf_iterations: int = 25
    
    # Parallelization
    default_workers: int = -1  # -1 = auto-detect (CPU count - 1)
    
    # NSGA-II defaults
    default_population: int = 100
    default_generations: int = 50
    default_crossover_prob: float = 0.9
    
    # Representative days for fast optimization
    default_representative_days: int = 12
    
    # Cache precision for evaluation cache keys
    cache_precision: int = 6  # decimal places for cache key rounding


# ======================== CAPACITY DEFAULTS ========================

@dataclass(frozen=True)
class CapacityConfig:
    """Default BESS capacity parameters."""
    default_capacity_mwh: float = 1.0
    default_capacity_min: float = 0.5
    default_capacity_max: float = 10.0


# ======================== LOAD GROWTH PARAMETERS ========================

@dataclass(frozen=True)
class LoadGrowthConfig:
    """Load growth modeling parameters."""
    annual_growth_rate: float = 0.02  # 2% annual growth
    model_years: int = None           # None = use project_life


# ======================== FILE PATHS ========================

@dataclass(frozen=True)
class FileConfig:
    """Default file paths and names."""
    default_input_csv: str = "Data_2024_33bus_scaled_fullyear.csv"
    default_output_dir: str = "optimization_results"
    results_json: str = "optimization_results.json"
    pareto_csv: str = "pareto_solutions.csv"
    all_solutions_csv: str = "all_pareto_solutions.csv"
    pareto_plot_2d: str = "pareto_front_2d.png"
    pareto_plot_3d: str = "pareto_front_3d.png"
    convergence_plot: str = "convergence.png"


# ======================== MAGIC NUMBERS & THRESHOLDS ========================

@dataclass(frozen=True)
class ThresholdConfig:
    """Various threshold values used in calculations."""
    min_annual_energy_mwh: float = 0.01
    min_energy_threshold_gwp: float = 1.0
    lcos_sentinel_value: float = 999999.0
    gwp_sentinel_value: float = 9999.0
    heuristic_voltage_threshold: float = 0.8
    heuristic_voltage_scale: float = 0.08
    heuristic_bus_voltage_fraction: float = 0.2
    epsilon: float = 1e-9  # Small value to avoid division by zero


# ======================== RESULT INDICES ========================

# Result array indices for evaluate_objectives_simple return value
# result = [capacity, F1_NPV, F2_fluctuation, F3_voltage_total, F3_voltage_bus, F4_GWP, LCOS]
RESULT_IDX_CAPACITY = 0
RESULT_IDX_NPV = 1
RESULT_IDX_FLUCTUATION = 2
RESULT_IDX_VOLTAGE_TOTAL = 3
RESULT_IDX_VOLTAGE_BUS = 4
RESULT_IDX_GWP = 5
RESULT_IDX_LCOS = 6


# ======================== GLOBAL INSTANCES ========================
# Create singleton instances for easy access throughout the codebase

BESS_CONFIG = BessConfig()
FINANCIAL_CONFIG = FinancialConfig()
SOURCE_CONFIG = SourceConfig()
ENVIRONMENTAL_CONFIG = EnvironmentalConfig()
VOLTAGE_CONFIG = VoltageConfig()
NETWORK_CONFIG = NetworkConfig()
OPTIMIZATION_CONFIG = OptimizationConfig()
CAPACITY_CONFIG = CapacityConfig()
LOAD_GROWTH_CONFIG = LoadGrowthConfig()
FILE_CONFIG = FileConfig()
THRESHOLD_CONFIG = ThresholdConfig()


# ======================== LEGACY COMPATIBILITY ========================
# Maintain backward compatibility with old dict-based constants

BESS_CONSTRAINTS = {
    "c_rate": BESS_CONFIG.c_rate,
    "rte": BESS_CONFIG.rte,
    "soc_min": BESS_CONFIG.soc_min,
    "soc_max": BESS_CONFIG.soc_max,
    "initial_soc": BESS_CONFIG.initial_soc,
}

FINANCIAL_PARAMS = {
    "capex_per_mwh": FINANCIAL_CONFIG.capex_per_mwh,
    "om_per_mwh": FINANCIAL_CONFIG.om_per_mwh,
    "discount_rate": FINANCIAL_CONFIG.discount_rate,
    "project_life": FINANCIAL_CONFIG.project_life,
    "lifetime_factor": FINANCIAL_CONFIG.lifetime_factor,
}

SOURCE_PARAMS = {
    "cost_conventional": SOURCE_CONFIG.cost_conventional,
    "Te_E": SOURCE_CONFIG.te_e,
}

ENV_PARAMS = {
    "emb_gwp_per_mwh_cap": ENVIRONMENTAL_CONFIG.emb_gwp_per_mwh_cap,
    "recycling_credit_factor": ENVIRONMENTAL_CONFIG.recycling_credit_factor,
    "eol_transport_per_mwh": ENVIRONMENTAL_CONFIG.eol_transport_per_mwh,
}

VOLTAGE_LIMITS = {
    "min_pu": VOLTAGE_CONFIG.min_pu,
    "max_pu": VOLTAGE_CONFIG.max_pu,
}

LOAD_GROWTH_PARAMS = {
    "annual_growth_rate": LOAD_GROWTH_CONFIG.annual_growth_rate,
    "model_years": LOAD_GROWTH_CONFIG.model_years,
}

# Simple constants
DEFAULT_NETWORK = NETWORK_CONFIG.default_network
DEFAULT_CAPACITY_MWH = CAPACITY_CONFIG.default_capacity_mwh
DEFAULT_BUS_INDEX = NETWORK_CONFIG.default_bus_index
DEFAULT_TIMESTEP_HOURS = NETWORK_CONFIG.default_timestep_hours
DEFAULT_INPUT_CSV = FILE_CONFIG.default_input_csv


def get_all_configs() -> Dict[str, Any]:
    """
    Get all configuration objects as a dictionary.
    
    Returns:
        Dict mapping config names to config objects
    """
    return {
        'bess': BESS_CONFIG,
        'financial': FINANCIAL_CONFIG,
        'source': SOURCE_CONFIG,
        'environmental': ENVIRONMENTAL_CONFIG,
        'voltage': VOLTAGE_CONFIG,
        'network': NETWORK_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'capacity': CAPACITY_CONFIG,
        'load_growth': LOAD_GROWTH_CONFIG,
        'file': FILE_CONFIG,
        'threshold': THRESHOLD_CONFIG,
    }


def print_configuration():
    """Print all configuration parameters for debugging."""
    configs = get_all_configs()
    print("=" * 70)
    print("BESS OPTIMIZATION CONFIGURATION")
    print("=" * 70)
    for name, config in configs.items():
        print(f"\n{name.upper()}:")
        for field, value in config.__dict__.items():
            print(f"  {field}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    print_configuration()
