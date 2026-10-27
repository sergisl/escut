import os
import sys
import warnings

# Ensure the refactored modules in `main/` are importable as top-level modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Silence noisy upstream deprecation warnings from matplotlib during tests
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"matplotlib.*",
)

# The package intentionally emits a deprecation warning about re-exports;
# tests don't need to show it every run.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Qsolver re-exports modular APIs.*",
)
