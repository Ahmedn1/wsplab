"""wsplab: weight-space programming, end to end."""

__version__ = "0.1.0.dev0"
__all__ = ["RunResult", "run"]


def __getattr__(name: str):
    """Import the full stack only when it is used: patch tools and the sandbox need numpy alone."""
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
