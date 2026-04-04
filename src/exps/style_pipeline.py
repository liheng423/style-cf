"""
Compatibility shim.

Preferred import path: ``src.style_pipeline``.
"""

from ..style_pipeline import (
    StyleCaseResult,
    StyleNotebookPipeline,
    SimpleNamespace,
    main,
    run_style_pipeline_from_config,
)

__all__ = [
    "StyleCaseResult",
    "StyleNotebookPipeline",
    "SimpleNamespace",
    "run_style_pipeline_from_config",
    "main",
]
