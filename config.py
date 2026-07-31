from pathlib import Path

def get_base_dir():
    """Find the project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # If not found, return the directory of this config file (as a fallback)
    return current.parent

BASE_DIR = get_base_dir()
DB_NAME = BASE_DIR / "db" / "local_market_share.db"