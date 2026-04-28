"""Compatibility shim for legacy ``rsl_rl.env`` imports.

The bundled ``legged_gym`` code imports ``VecEnv`` only for type annotations,
but this repository's ``rsl_rl`` snapshot does not ship the original
``rsl_rl.env`` module. A lightweight placeholder keeps those imports working
without changing runtime behavior.
"""


class VecEnv:
    """Marker base used only for annotations in this repository."""

    pass
