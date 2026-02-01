import sys
from pathlib import Path

try:
    import rootutils  # type: ignore
except ModuleNotFoundError:
    rootutils = None


def _fallback_root(start_file: str) -> Path:
    """Find repo root by walking up until .project-root is found."""
    path = Path(start_file).resolve()
    for parent in [path] + list(path.parents):
        if (parent / ".project-root").exists():
            return parent
    return path.parent


def pytest_configure(config):
    if rootutils is not None:
        root = rootutils.setup_root(
            __file__, indicator=".project-root", pythonpath=True, dotenv=True
        )
    else:
        root = _fallback_root(__file__)
        # mimic pythonpath=True behavior
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(root / "models" / "rfd3" / "tests"))

    paths_to_add = [
        root / "src",
        root / "models" / "rfd3" / "tests",
        root / "atomworks" / "src",
    ]

    for path in paths_to_add:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Add markers
    config.addinivalue_line("markers", "fast: mark test as fast (run quickly)")
    config.addinivalue_line("markers", "slow: mark test as slow (run slowly)")
