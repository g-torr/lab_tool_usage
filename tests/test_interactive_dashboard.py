import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interactive_dashboard import analyze_data, resolve_db_path


def test_resolve_db_path_points_to_repository_database():
    db_path = resolve_db_path()

    assert db_path.exists()
    assert db_path.name == "local_market_share.db"


def test_analyze_data_returns_non_empty_dataframe():
    df = analyze_data()

    assert df is not None
    assert not df.empty
    assert {"year_month", "resolved_machine", "mention_count"}.issubset(df.columns)
