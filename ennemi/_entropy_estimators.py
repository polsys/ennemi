# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""The estimator methods.

Do not import this module directly.
Use the `estimate_mi` method in the main ennemi module instead.

This module selects between the pure Python and Numba implementation at runtime.
"""

# Import the Numba implementation by default
# This will fail if Numba is not installed or is disabled via environment variable
try:
    from ._entropy_estimators_numba import _estimate_single_mi,\
        _estimate_single_entropy_1d, _psi

    # Import some methods from the pure Python version
    # as long as the N-dimensional grid code is not optimized
    from ._entropy_estimators_pure import _estimate_single_entropy_nd,\
        _estimate_conditional_mi
        
except ImportError:
    from ._entropy_estimators_pure import _estimate_single_mi,\
        _estimate_conditional_mi,\
        _estimate_single_entropy_1d, _estimate_single_entropy_nd, _psi

__all__ = [
    "_estimate_single_mi",
    "_estimate_conditional_mi",
    "_estimate_single_entropy_1d",
    "_estimate_single_entropy_nd",
    "_psi"
]
