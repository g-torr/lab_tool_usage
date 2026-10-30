import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.main import fetch_methods_text_from_web


@pytest.mark.parametrize("heading", ["METHOD DETAILS", "Method"])
def test_fetch_methods_text_accepts_method_heading_variants(heading):
    method_body = "Experimental protocol. " * 20
    response = Mock(
        status_code=200,
        text=(
            f"<html><body><div class='section'><h2>{heading}</h2>"
            f"<div class='subsection'><p>{method_body}</p></div></div>"
            "<h2>REFERENCES</h2></body></html>"
        ),
        headers={},
    )
    session = Mock()
    session.get.return_value = response

    with patch("src.main.requests.Session", return_value=session):
        methods_text = fetch_methods_text_from_web("10.1101/2024.07.15.603616", version=1)

    assert methods_text is not None
    assert "experimental protocol" in methods_text
    session.get.assert_called_once()
