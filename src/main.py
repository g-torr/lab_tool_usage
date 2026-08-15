import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import logging
import psycopg2
import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATABASE_URL = os.getenv("DATABASE_URL")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set!")
    return psycopg2.connect(DATABASE_URL)

# ==========================================
# CONFIGURATION & FILTERS
# ==========================================
TARGET_CATEGORIES = ["genomics", "bioinformatics", "cell_biology", "molecular_biology", 
                     "cancer_biology", "neuroscience", "immunology", "developmental_biology", "systems_biology"]

BROAD_CATCHMENT = [
    "spatial transcriptomics", "visium", "xenium", "merscope", "merfish", "seqfish", "cosmx", "stereo-seq", "phenocycler",
    "single-cell", "scrna-seq", "chromium", "flow cytometry", "facs", "cytof", 
    "mass spectrometry", "lc-ms", "orbitrap", "timstof", "maldi", 
    "cryo-em", "cryo-et", "super-resolution", "confocal", "light-sheet",
    "novaseq", "pacbio", "nanopore", "promethion", "hifi"
]

EXCLUSION_TERMS = ["review", "perspective", "commentary", "meta-analysis", "systematic review", "benchmark", "re-analysis"]
DRY_LAB_TRIGGERS = ["public database", "publicly available", "geo", "sra", "arrayexpress", "downloaded from", "re-analysis", "in silico"]

INITIAL_SCAN_DATE = date(2023, 1, 1)
SCAN_OVERLAP_DAYS = 1

# ==========================================
# DATABASE INITIALIZATION & RESET
# ==========================================
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            doi TEXT PRIMARY KEY,
            title TEXT,
            category TEXT,
            date TEXT,
            abstract TEXT,
            processed INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machine_mentions (
            id SERIAL PRIMARY KEY,
            doi TEXT REFERENCES candidates(doi),
            raw_llm_machine TEXT,
            resolved_machine TEXT,
            parent_company TEXT,
            ticker TEXT,
            confidence REAL,
            is_novel INTEGER DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id BIGSERIAL PRIMARY KEY,
            window_start DATE NOT NULL,
            window_end DATE NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            error TEXT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def reset_db():
    """Replaces SQLite os.remove by dropping existing PostgreSQL tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    logger.info("Resetting PostgreSQL database tables...")
    cursor.execute("DROP TABLE IF EXISTS machine_mentions CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS candidates CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS ingestion_runs CASCADE;")
    conn.commit()
    cursor.close()
    conn.close()
    init_db()

# ==========================================
def next_scan_window():
    """Return the next scan window after the latest successful run."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT window_end
        FROM ingestion_runs
        WHERE status = 'completed'
        ORDER BY window_end DESC, completed_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    today = datetime.now(timezone.utc).date()
    start = INITIAL_SCAN_DATE if row is None else row[0] - timedelta(days=SCAN_OVERLAP_DAYS)
    return start, today

def begin_ingestion_run(start, end):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ingestion_runs (window_start, window_end, status)
        VALUES (%s, %s, 'running')
        RETURNING id
    """, (start, end))
    run_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return run_id

def finish_ingestion_run(run_id, status, error=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE ingestion_runs
        SET status = %s,
            completed_at = CASE WHEN %s = 'completed' THEN NOW() ELSE completed_at END,
            error = %s
        WHERE id = %s
    """, (status, status, error, run_id))
    conn.commit()
    cursor.close()
    conn.close()


# STAGE 1 & 2: HARVESTING
# ==========================================
def pipeline_stage_1_and_2(start_date, end_date):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    cursor_idx = 0
    
    logger.info(f"\n--> Commencing harvest from {start_date} to {end_date}...")
    while True:
        url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor_idx}/json"
        try:
            response = session.get(url, timeout=30)
            if response.status_code != 200:
                cursor_idx += 100
                continue
            data = response.json()
            papers = data.get("collection", [])
            if not papers: break
            
            batch_candidates = []
            for paper in papers:
                paper_category = paper.get("category", "").lower()
                if paper_category in TARGET_CATEGORIES:
                    title = paper.get("title", "").lower()
                    abstract = paper.get("abstract", "").lower()
                    doi = paper.get("doi")
                    
                    if any(term in title or term in abstract for term in EXCLUSION_TERMS): continue
                    if any(word in abstract or word in title for word in BROAD_CATCHMENT):
                        batch_candidates.append((doi, paper.get("title"), paper_category, paper.get("date"), paper.get("abstract")))
            
            if batch_candidates:
                cursor.executemany("""
                    INSERT INTO candidates (doi, title, category, date, abstract)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (doi) DO NOTHING
                """, batch_candidates)
                conn.commit()
                logger.info(f"   Saved {len(batch_candidates)} candidates.")
            cursor_idx += 100
            time.sleep(0.4)
        except Exception as e:
            logger.error(f"Exception at {cursor_idx}: {e}")
            cursor_idx += 100
            continue
            
    cursor.close()
    conn.close()

# ==========================================
# WEB SCRAPER
# ==========================================
def fetch_methods_text_from_web(doi):
    session = requests.Session()
    retries = Retry(
        total=5, 
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    heading_weights = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4}
    stop_terms = ["reference", "acknowledgement", "conflict of interest", "funding", "author contribution"]
    base_domains = ["https://www.biorxiv.org/content", "https://www.medrxiv.org/content"]
    
    for base_url in base_domains:
        web_url = f"{base_url}/{doi}.full"
        try:
            response = session.get(web_url, headers=headers, timeout=30)
            logger.info(f"Fetching URL: {web_url} - Status Code: {response.status_code}")
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning(f"Rate limited (429). Sleeping for {retry_after} seconds...")
                time.sleep(retry_after)
                response = session.get(web_url, headers=headers, timeout=30)
                
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
                
                for h in headings:
                    heading_text = h.get_text(strip=True).lower()
                    if "method" in heading_text or "materials and methods" in heading_text:
                        target_tag = h.name
                        target_weight = heading_weights[target_tag]
                        
                        fragments = []
                        for sibling in h.find_next_siblings():
                            if sibling.name in heading_weights:
                                sibling_text = sibling.get_text(strip=True).lower()
                                sibling_weight = heading_weights[sibling.name]
                                
                                if sibling_weight <= target_weight:
                                    break
                                if any(term in sibling_text for term in stop_terms):
                                    break
                            
                            fragments.append(sibling.get_text(separator=' ', strip=True))
                        
                        combined_text = " ".join(fragments).strip().lower()
                        if len(combined_text) > 200:
                            logger.info(f"Method section successfully extracted from {base_url}.")
                            return combined_text
                
                logger.warning(f"Page fetched from {base_url}, but no valid Method boundaries were found in DOM.")
            else:
                logger.warning(f"Failed to fetch {web_url} - Status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Exception occurred while fetching {web_url}: {e}")
            continue
            
    return None

# ==========================================
# STAGE 3: LLM + SEMANTIC RESOLUTION
# ==========================================
def pipeline_stage_3_execution():
    from novelty_classifier import classify_novelty
    from semantic_resolver import SemanticMachineResolver
    from machine_extractor import (
        extract_machines_hybrid,
        load_registry_gazetteer,
        load_stage2_labels,
    )

    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("Loading Candidate Registry Gazetteer and Stage 2 Labels...")
    load_registry_gazetteer()
    target_labels = load_stage2_labels()
    
    print("Initializing Semantic Machine Resolver...")
    resolver = SemanticMachineResolver()
    
    cursor.execute("SELECT doi, abstract FROM candidates WHERE processed = 0")
    unprocessed = cursor.fetchall()
    print(f"Found {len(unprocessed)} unprocessed candidates.")
    
    for doi, abstract in unprocessed:
        print(f"\n{'='*70}")
        print(f"Processing: {doi}")
        print(f"{'='*70}")
        
        methods_text = fetch_methods_text_from_web(doi)
        time.sleep(1.0)
        
        if not methods_text:
            print(f"   ❌ Skipping: No methods text found.")
            cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = %s", (doi,))
            conn.commit()
            continue
        
        # STAGE 1: Novelty Classification
        print(f"\n   [STAGE 1/3] Classifying novelty...")
        novelty_result = classify_novelty(abstract, methods_text[:2000])
        
        if not novelty_result.is_novel_experiment:
            print(f"   ❌ [NON-NOVEL] {novelty_result.reasoning}")
            cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = %s", (doi,))
            conn.commit()
            continue
        
        print(f"   ✅ [NOVEL] {novelty_result.reasoning}")
        
        # STAGE 2: Hybrid Machine Extraction
        print(f"\n   [STAGE 2/3] Extracting candidates from {len(methods_text)} chars...")
        raw_machines = extract_machines_hybrid(methods_text, target_labels=target_labels)
        
        if not raw_machines:
            print(f"   ⚠️  [NO MACHINES] No equipment mentioned in methods.")
            cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = %s", (doi,))
            conn.commit()
            continue
        
        print(f"   ✅ Found {len(raw_machines)} candidate machine mentions.")
        
        # STAGE 3: Semantic Resolution
        print(f"\n   [STAGE 3/3] Resolving {len(raw_machines)} candidates...")
        for raw_machine in raw_machines:
            resolved_name, parent, ticker, conf = resolver.resolve(raw_machine)
            
            cursor.execute("""
                INSERT INTO machine_mentions 
                (doi, raw_llm_machine, resolved_machine, parent_company, ticker, confidence, is_novel) 
                VALUES (%s, %s, %s, %s, %s, %s, 1)
            """, (doi, raw_machine, resolved_name, parent, ticker, conf))
            
            if resolved_name:
                status = f"→ {resolved_name} ({parent} / {ticker}) [conf: {conf:.2f}]"
                print(f"      ✅ {raw_machine} {status}")
            else:
                status = f"→ UNRESOLVED [conf: {conf:.2f}]"
                print(f"      ⚠️  {raw_machine} {status}")
        
        cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = %s", (doi,))
        conn.commit()

    cursor.close()
    conn.close()
    print(f"\n{'='*70}")
    print("Pipeline execution complete.")
    print(f"{'='*70}")

# ==========================================
# EXECUTION ENTRYPOINT
# ==========================================
def build_parser():
    parser = argparse.ArgumentParser(description='Run the semantic pipeline for spatial transcriptomics equipment tracking.')
    parser.add_argument(
        '--append',
        '--no-reset',
        dest='append',
        action='store_true',
        help='Append newly discovered papers to the existing database without clearing it first.',
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help='Optional explicit first publication date (YYYY-MM-DD); otherwise use the database checkpoint.',
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='Optional explicit last publication date (YYYY-MM-DD); otherwise use today and the database checkpoint.',
    )
    return parser

if __name__ == "__main__":
    from create_registry import create_registry

    parser = build_parser()
    args = parser.parse_args()

    logger.info("Executing Semantic Pipeline Architecture...")
    create_registry()

    if args.append:
        logger.info("Append mode enabled: preserving existing database entries.")
        init_db()
    else:
        reset_db()

    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be provided together")

    if args.start_date is None:
        scan_start, scan_end = next_scan_window()
    else:
        scan_start = date.fromisoformat(args.start_date)
        scan_end = date.fromisoformat(args.end_date)

    if scan_start > scan_end:
        logger.info("No new scan window is available through today.")
        raise SystemExit(0)

    run_id = begin_ingestion_run(scan_start, scan_end)
    try:
        pipeline_stage_1_and_2(scan_start.isoformat(), scan_end.isoformat())
        pipeline_stage_3_execution()
    except Exception as exc:
        finish_ingestion_run(run_id, "failed", str(exc))
        logger.exception("Ingestion run %s failed; its window will be retried.", run_id)
        raise
    else:
        finish_ingestion_run(run_id, "completed")
