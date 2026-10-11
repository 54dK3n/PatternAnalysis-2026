"""Audited ADNI splits, manifests, slice transforms, and data loaders."""


def __getattr__(name: str) -> object:
    """Import loaders lazily so the torch-free source audit can import this package."""
    if name == "make_loader":
        from .loaders import make_loader
        return make_loader
    if name in ("load_fold", "load_holdout"):
        from . import manifests
        return getattr(manifests, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["make_loader", "load_fold", "load_holdout"]
