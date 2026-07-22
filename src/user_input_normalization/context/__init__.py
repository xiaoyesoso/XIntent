"""Deep context integration module (corresponding to task 7 / D14).

Provides unified assembly capability for three layers of context
(user profile + key facts + dialogue history recall),
supporting pre-normalization and deep-normalization inference.
"""

from .integrator import ContextBundle, ContextIntegrator

__all__ = ["ContextBundle", "ContextIntegrator"]
