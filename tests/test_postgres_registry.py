import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.create_registry import create_registry
from src.machine_extractor import load_stage2_labels, load_registry_gazetteer
from src.db import get_db_connection


def cleanup_registry_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS registry_machines CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS stage2_labels CASCADE;")
    conn.commit()
    cursor.close()
    conn.close()


def test_create_registry_populates_postgres():
    cleanup_registry_tables()

    create_registry()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM registry_machines")
    registry_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM stage2_labels")
    labels_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    assert registry_count > 0
    assert labels_count > 0


def test_load_stage2_labels_reads_from_postgres():
    labels = load_stage2_labels()
    assert isinstance(labels, list)
    assert len(labels) > 0
    assert "dna sequencer" in labels


def test_load_registry_gazetteer_populates_keywords():
    import src.machine_extractor as machine_extractor

    # Ensure registry tables exist and are populated before testing the gazetteer.
    create_registry()

    load_registry_gazetteer()

    matches = machine_extractor.keyword_processor.extract_keywords("NovaSeq X / 6000")
    assert "NovaSeq X / 6000" in matches

    matches_alias = machine_extractor.keyword_processor.extract_keywords("illumina novaseq")
    assert "NovaSeq X / 6000" in matches_alias
