import json
import logging
import os
import re
from typing import List, Set
import pandas as pd
from flashtext import KeywordProcessor
from gliner import GLiNER
import torch
from src.db import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global FlashText Processor for fast exact matching
keyword_processor = KeywordProcessor(case_sensitive=False)


def reset_registry_gazetteer():
    global keyword_processor
    keyword_processor = KeywordProcessor(case_sensitive=False)

# Local fallback labels in case the database is unavailable
DEFAULT_STAGE2_LABELS = [
    "dna sequencer",
    "single cell controller",
    "microscope",
    "cell sorter",
    "flow cytometer",
    "mass spectrometer",
    "electrophysiology equipment",
    "chromatography system",
    "single cell isolation system"
]


def load_stage2_labels() -> List[str]:
    """Loads GLiNER target labels from PostgreSQL."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT label FROM stage2_labels")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        labels = [row[0] for row in rows]
        if labels:
            logger.info(f"Loaded {len(labels)} extraction labels from PostgreSQL")
            return labels
    except Exception as e:
        logger.warning(f"Failed to load stage2_labels from PostgreSQL: {e}")
    
    logger.warning("Falling back to default stage2 labels.")
    return DEFAULT_STAGE2_LABELS


def load_registry_gazetteer():
    """Populates the FlashText gazetteer with candidate machine aliases from PostgreSQL."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT canonical_name, aliases FROM registry_machines", conn)
        conn.close()
    except Exception as e:
        logger.error(f"Failed to populate gazetteer from PostgreSQL: {e}")
        return

    count = 0
    for _, row in df.iterrows():
        canonical = row['canonical_name']
        if pd.notna(canonical):
            keyword_processor.add_keyword(canonical, canonical)
            count += 1

        if pd.notna(row.get('aliases')):
            aliases = json.loads(row['aliases']) if isinstance(row['aliases'], str) else row['aliases']
            for alias in aliases:
                if len(alias) >= 3:
                    keyword_processor.add_keyword(alias, canonical)
                    count += 1

    logger.info(f"Loaded {count} candidate keywords into gazetteer from PostgreSQL.")

# High-frequency noise terms to ignore during zero-shot NER
NOISE_TERMS = {
    "plate", "plates", "well", "wells", "objective", "lens", 
    "buffer", "kit", "reagent", "tube", "tubes", "dish", "coverslip",
    "slide", "slides", "chamber", "filter", "filters", "membrane",
    "cells", "clone", "clones", "pcr", "qpcr", "scrna-seq", "scrnaseq"
}

# -----------------------------------------------------------------------------
# 2. MODEL INITIALIZATION
# -----------------------------------------------------------------------------
logger.info("Initializing GLiNER-2.1 Large model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
ner_model = GLiNER.from_pretrained("urchade/gliner_large-v2.1").to(device)

# -----------------------------------------------------------------------------
# 3. EXTRACTION & POST-PROCESSING
# -----------------------------------------------------------------------------
import re

# 1. Regex for biological/database accessions that mimic model numbers
ACCESSION_PATTERNS = [
    re.compile(r'\bgse\d+\b', re.IGNORECASE),      # GEO Datasets
    re.compile(r'\bsrx\d+\b', re.IGNORECASE),      # SRA
    re.compile(r'\buniprot\b', re.IGNORECASE),     # Protein DB
    re.compile(r'\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b'), # UniProt Accession format
    re.compile(r'\b[A-N,R-Z][0-9][A-Z][A-Z0-9]{2}[0-9]\b') 
]

# 2. Strict blocklist for software, consumables, and standalone vendors
STRICT_BLOCKLIST = {
    "deseq2", "dorado", "cellranger", "seurat", "scanpy", "maxquant",
    "cellvis", "invitrogen", "polysciences", "falcon", "ficoll", 
    "thermo fisher", "oxford nanopore technologies", "agilent technologies"
}

def post_filter_entities(entities: List[dict]) -> List[str]:
    cleaned = []
    for ent in entities:
        text = ent["text"].strip()
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))
        
        #  noise filter
        if words.intersection(NOISE_TERMS):
            continue
            
        #  Block standalone vendors and software
        if text_lower in STRICT_BLOCKLIST:
            continue
            
        #  Block biological cohorts (e.g., "HIV controllers")
        if "controllers" in text_lower and "chromium" not in text_lower:
            continue
            
        #  Block database accessions
        if any(pattern.search(text) for pattern in ACCESSION_PATTERNS):
            continue
            
        if len(text) < 4:
            continue
            
        cleaned.append(text)
    return sorted(list(set(cleaned)))

def extract_machines_hybrid(methods_text: str, target_labels: List[str] = None, threshold: float = 0.40) -> List[str]:
    """
    Combines direct candidate matching (Gazetteer) with zero-shot NER (GLiNER).
    """
    if not methods_text or not methods_text.strip():
        return []

    if target_labels is None:
        target_labels = load_stage2_labels()

    extracted_candidates: Set[str] = set()
    
    # 1. Direct candidate matching (Fast & Exact)
    gazetteer_matches = keyword_processor.extract_keywords(methods_text)
    extracted_candidates.update(gazetteer_matches)
    
    # 2. GLiNER NER extraction (Fallback for unlisted/novel models)
    ner_entities = ner_model.predict_entities(
        methods_text, 
        target_labels, 
        threshold=threshold,
        flat_ner=True
    )
    cleaned_ner = post_filter_entities(ner_entities)
    extracted_candidates.update(cleaned_ner)
                
    return sorted(list(extracted_candidates))