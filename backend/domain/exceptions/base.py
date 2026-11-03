"""Root of the project exception hierarchy."""

from __future__ import annotations


class FedSecureAIError(Exception):
    """Base class for every error raised deliberately by this project.

    Callers can catch this to distinguish an expected, explained failure from
    an unexpected bug. Low-level third-party exceptions are translated into
    subclasses of this type at the infrastructure boundary, so upper layers
    never have to know that (for example) a Kaggle client raised an HTTP error.
    """
