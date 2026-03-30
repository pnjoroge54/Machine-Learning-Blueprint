"""
CV Convenience Layer (thin shim over unified @cacheable)
"""

from .unified_cache import cacheable


# Backwards compatibility shim
def cv_cacheable(*args, **kwargs):
    """Old decorator name → new unified decorator."""
    if args and callable(args[0]):
        return cacheable(args[0], **kwargs)
    return cacheable(*args, **kwargs)


# Example usage (add your CV functions here if desired)
@cacheable(time_aware=True, auto_versioning=True)
def clf_hyper_fit(data, clf, param_distributions, **kwargs):
    # Your CV logic here
    pass


__all__ = ["cv_cacheable", "clf_hyper_fit"]
