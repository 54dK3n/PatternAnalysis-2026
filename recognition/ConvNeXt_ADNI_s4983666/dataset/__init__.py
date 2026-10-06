"""Audited ADNI splits, manifests, slice transforms, and data loaders."""


def __getattr__(name: str) -> object:
    """Expose course-facing loaders lazily, keeping source audits torch-free."""
    if name == "make_loader":
        from .loaders import make_loader
        return make_loader
    if name == "load_fold":
        from .manifests import load_fold
        return load_fold
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["make_loader", "load_fold"]
