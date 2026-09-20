from unittest.mock import Mock

from src.main import fetch_first_version_metadata


def test_first_version_metadata_uses_earliest_version_and_its_date():
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {
        "collection": [
            {
                "doi": "10.1101/2024.02.20.581234",
                "version": "3",
                "date": "2024-06-01",
                "title": "Revised title",
                "category": "genomics",
                "abstract": "revised abstract",
            },
            {
                "doi": "10.1101/2024.02.20.581234",
                "version": "1",
                "date": "2024-02-21",
                "title": "Original title",
                "category": "genomics",
                "abstract": "original abstract",
            },
        ]
    }
    session.get.return_value = response

    result = fetch_first_version_metadata(
        session, "10.1101/2024.02.20.581234", "biorxiv"
    )

    assert result["version"] == 1
    assert result["date"] == "2024-02-21"
    assert result["title"] == "Original title"
    assert result["source_version_url"].endswith("581234v1.full")
    assert result["first_version_api_url"].endswith("581234/na/json")


def test_first_version_metadata_rejects_unknown_servers():
    session = Mock()

    try:
        fetch_first_version_metadata(session, "10.1101/example", "unknown")
    except ValueError as exc:
        assert "Unsupported preprint server" in str(exc)
    else:
        raise AssertionError("unknown preprint server should fail")
