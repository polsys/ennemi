# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""The estimator methods.

Do not import this module directly.
Use the `estimate_mi` method in the main ennemi module instead.

This module selects between the pure Python and Numba implementation at runtime.
"""

# Import the Numba implementation by default
try:
    from ._entropy_estimators_numba import _estimate_single_mi,\
        _estimate_conditional_mi,\
        _estimate_single_entropy_1d, _estimate_single_entropy_nd
        
except ImportError:
    from ._entropy_estimators_pure import _estimate_single_mi,\
        _estimate_conditional_mi,\
        _estimate_single_entropy_1d, _estimate_single_entropy_nd

__all__ = [
    "_estimate_single_mi",
    "_estimate_conditional_mi",
    "_estimate_single_entropy_1d",
    "_estimate_single_entropy_nd"
]