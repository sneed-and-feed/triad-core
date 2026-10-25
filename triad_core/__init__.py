"""
triad-core: Standalone, zero-dependency, proof-carrying Lean 4 + #![no_std] Rust SIMD + Python FFI library
unifying 6 core mathematical primitives across the DarkSnail constellation.
"""

from triad_core.operators import (
    TriadCoreEngine,
    # Canonical DarkSnail names
    skew_triad_step,
    slow_fast_cascade_k16,
    cayley_rotor_so16_step,
    beltrami_morse_gate,
    compute_spatial_mad_floor_256,
    kelvin_mandel_so6_rotor,
    rotate_stiffness_so6,
    contact_sheaf_se2_step,
    continuity_clutter_filter,
    # Stealth domain-native aliases
    so_skew_tiebreaker,
    spherical_cascade_integrator,
    cayley_attitude_update,
    morse_innovation_gate,
    anisotropic_stiffness_rotor,
    rigid_contact_coboundary_solver,
)

__version__ = "0.1.0"
__license__ = "CC0-1.0"

__all__ = [
    "TriadCoreEngine",
    "skew_triad_step",
    "so_skew_tiebreaker",
    "slow_fast_cascade_k16",
    "spherical_cascade_integrator",
    "cayley_rotor_so16_step",
    "cayley_attitude_update",
    "beltrami_morse_gate",
    "morse_innovation_gate",
    "compute_spatial_mad_floor_256",
    "kelvin_mandel_so6_rotor",
    "anisotropic_stiffness_rotor",
    "rotate_stiffness_so6",
    "contact_sheaf_se2_step",
    "rigid_contact_coboundary_solver",
    "continuity_clutter_filter",
    "__version__",
]
