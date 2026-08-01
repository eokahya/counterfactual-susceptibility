"""Project-specific exceptions with actionable scientific diagnostics."""

from __future__ import annotations


class CfsusError(Exception):
    """Base class for expected Counterfactual Susceptibility errors."""


class ScientificInputError(CfsusError, ValueError):
    """Raised when a scientific input violates the declared convention."""


class NonFiniteInputError(ScientificInputError):
    """Raised when a scientific scalar is NaN or infinite."""


class ActiveTargetError(ScientificInputError):
    """Raised when an inactive-target calculation receives an active target."""


class BackendError(CfsusError, RuntimeError):
    """Base class for backend integration failures."""


class BackendUnavailableError(BackendError):
    """Raised when an optional backend dependency is not installed."""


class UnsupportedBackendOperationError(BackendError, NotImplementedError):
    """Raised when a backend operation is unavailable or not yet verified."""
