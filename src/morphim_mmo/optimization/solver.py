"""JAX-autodifferentiated Levenberg--Marquardt wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SolveResult:
    """Backend-independent least-squares result fields used by the pipeline."""

    parameters: np.ndarray
    cost: float
    optimality: float
    evaluations: int
    success: bool
    message: str


def _compile_scipy_callbacks(
    residual_function: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Compile one JAX residual and exact Jacobian for SciPy least-squares."""

    residual_jit = jax.jit(residual_function)
    jacobian_jit = jax.jit(jax.jacfwd(residual_function))

    def fun(parameters: np.ndarray) -> np.ndarray:
        """Evaluate the compiled residual through SciPy's NumPy interface."""

        return np.asarray(residual_jit(jnp.asarray(parameters)), dtype=np.float64)

    def jac(parameters: np.ndarray) -> np.ndarray:
        """Evaluate the compiled exact Jacobian through SciPy's interface."""

        return np.asarray(jacobian_jit(jnp.asarray(parameters)), dtype=np.float64)

    return fun, jac


def solve_lm(
    residual_function: Callable[[jnp.ndarray], jnp.ndarray],
    initial_parameters: np.ndarray,
    max_evaluations: int = 200,
) -> SolveResult:
    """Solve a differentiable residual problem with LM and an exact JAX Jacobian."""

    fun, jac = _compile_scipy_callbacks(residual_function)

    result = least_squares(
        fun,
        np.asarray(initial_parameters, dtype=np.float64),
        jac=jac,
        method="lm",
        max_nfev=max_evaluations,
    )
    return SolveResult(
        parameters=result.x,
        cost=float(result.cost),
        optimality=float(result.optimality),
        evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def solve_bounded_least_squares(
    residual_function: Callable[[jnp.ndarray], jnp.ndarray],
    initial_parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    max_evaluations: int = 500,
) -> SolveResult:
    """Solve with an exact JAX Jacobian and finite parameter bounds."""

    fun, jac = _compile_scipy_callbacks(residual_function)

    result = least_squares(
        fun,
        np.asarray(initial_parameters, dtype=np.float64),
        jac=jac,
        method="trf",
        bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
        max_nfev=max_evaluations,
    )
    return SolveResult(
        parameters=result.x,
        cost=float(result.cost),
        optimality=float(result.optimality),
        evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )
