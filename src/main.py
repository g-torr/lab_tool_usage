import os
import sqlite3
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import logging
from config import DB_NAME

# Import from your local files (adjust 'lib.' prefix if they are in the same directory)
from semantic_resolver import SemanticMachineResolver
from create_registry import create_registry
from machine_extractor import (
    extract_machines_hybrid, 
    load_registry_gazetteer, 
    load_stage2_labels
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

# ==========================================
# DATABASE INITIALIZATION (NEW SCHEMA)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            doi TEXT PRIMARY KEY, title TEXT, category TEXT, date TEXT, abstract TEXT, processed INTEGER DEFAULT 0
        )
    """)
    # NEW SCHEMA: Captures raw LLM output, resolved entity, and corporate metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machine_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doi TEXT,
            raw_llm_machine TEXT,
            resolved_machine TEXT,
            parent_company TEXT,
            ticker TEXT,
            confidence REAL,
            is_novel INTEGER DEFAULT 0,
            FOREIGN KEY(doi) REFERENCES candidates(doi)
        )
    """)
    conn.commit()
    conn.close()

# ==========================================
# STAGE 1 & 2: HARVESTING
# ==========================================
def pipeline_stage_1_and_2(start_date, end_date):
    init_db()
    conn = sqlite3.connect(DB_NAME)
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
                cursor.executemany("INSERT OR IGNORE INTO candidates (doi, title, category, date, abstract) VALUES (?, ?, ?, ?, ?)", batch_candidates)
                conn.commit()
                logger.info(f"   Saved {len(batch_candidates)} candidates.")
            cursor_idx += 100
            time.sleep(0.4)
        except Exception as e:
            logger.error(f"Exception at {cursor_idx}: {e}")
            cursor_idx += 100
            continue
    conn.close()

# ==========================================
# WEB SCRAPER
# ==========================================
def fetch_methods_text_from_web(doi):
    # Establish a persistent session with robust retry logic for network flakes / 500s / 503s
    session = requests.Session()
    retries = Retry(
        total=5, 
        backoff_factor=2,  # Exponential backoff: 2s, 4s, 8s...
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Use a fully-formed User-Agent to stop triggering 403 WAF blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    # Structural hierarchy weights for heading depth validation
    heading_weights = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4}
    
    # Defensive backstop phrases to halt text harvesting if layout engines mis-tag end-matter sections
    stop_terms = ["reference", "acknowledgement", "conflict of interest", "funding", "author contribution"]
    
    base_domains = ["https://www.biorxiv.org/content", "https://www.medrxiv.org/content"]
    
    for base_url in base_domains:
        web_url = f"{base_url}/{doi}.full"
        try:
            # Increased timeout to 30s because full-text HTML packages can carry huge token payloads
            response = session.get(web_url, headers=headers, timeout=30)
            logger.info(f"Fetching URL: {web_url} - Status Code: {response.status_code}")
            
            # Handle rate-limiting dynamically
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
                        # Dynamically target the hierarchy node of this Methods container
                        target_tag = h.name
                        target_weight = heading_weights[target_tag]
                        
                        fragments = []
                        for sibling in h.find_next_siblings():
                            # Check if the sibling element encountered is another heading
                            if sibling.name in heading_weights:
                                sibling_text = sibling.get_text(strip=True).lower()
                                sibling_weight = heading_weights[sibling.name]
                                
                                # LAYER 1: Hierarchy Weight Gate
                                # Stop if we hit a heading of equivalent or greater structural importance (e.g., h2 -> h2)
                                if sibling_weight <= target_weight:
                                    break
                                    
                                # LAYER 2: Keyword Intercept Gate
                                # Fail-safe stop for flat/unstructured layouts where end-matter matches our blacklisted arrays
                                if any(term in sibling_text for term in stop_terms):
                                    break
                            
                            fragments.append(sibling.get_text(separator=' ', strip=True))
                        
                        combined_text = " ".join(fragments).strip().lower()
                        if len(combined_text) > 200:
                            logger.info(f"Method section successfully extracted using dual-layer boundary logic from {base_url}.")
                            return combined_text
                
                # Triggers only if the complete loop finishes without matching target keywords
                logger.warning(f"Page fetched successfully from {base_url}, but no valid Method boundaries were isolated in DOM.")
            else:
                logger.warning(f"Failed to fetch {web_url} - Server returned status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Exception occurred while fetching {web_url}: {e}")
            continue
            
    return None# ==========================================
# STAGE 3: LLM + SEMANTIC RESOLUTION
# ==========================================
from novelty_classifier import classify_novelty
from semantic_resolver import SemanticMachineResolver

def pipeline_stage_3_execution():
    """
    Three-stage pipeline:
    1. Novelty classification (fast, cheap)
    2. Hybrid Machine extraction (Gazetteer candidate matching + GLiNER)
    3. Semantic resolution (local, instant)
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Initialize gazetteer and target labels from candidate registry
    print("Loading Candidate Registry Gazetteer and Stage 2 Labels...")
    load_registry_gazetteer('machine_registry.csv')
    target_labels = load_stage2_labels()
    
    # 2. Initialize resolver once
    print("Initializing Semantic Machine Resolver...")
    resolver = SemanticMachineResolver('machine_registry.csv')
    
    cursor.execute("SELECT doi, abstract FROM candidates WHERE processed = 0")
    unprocessed = cursor.fetchall()
    print(f"Found {len(unprocessed)} unprocessed candidates.")
    
    for doi, abstract in unprocessed:
        print(f"\n{'='*70}")
        print(f"Processing: {doi}")
        print(f"{'='*70}")
        
        # Fetch methods
        methods_text = fetch_methods_text_from_web(doi)
        time.sleep(1.0)
        
        if not methods_text:
            print(f"   ❌ Skipping: No methods text found.")
            cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = ?", (doi,))
            conn.commit()
            continue
        
        # STAGE 1: Novelty Classification
        print(f"\n   [STAGE 1/3] Classifying novelty...")
        novelty_result = classify_novelty(abstract, methods_text[:2000])
        
        if not novelty_result.is_novel_experiment:
            print(f"   ❌ [NON-NOVEL] {novelty_result.reasoning}")
            cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = ?", (doi,))
            conn.commit()
            continue
        
        print(f"   ✅ [NOVEL] {novelty_result.reasoning}")
        
        # STAGE 2: Hybrid Machine Extraction (Gazetteer + GLiNER)
        print(f"\n   [STAGE 2/3] Extracting candidates from {len(methods_text)} chars...")
        
        # UPDATED CALL: Uses hybrid gazetteer + GLiNER extraction
        raw_machines = extract_machines_hybrid(methods_text, target_labels=target_labels)
        
        if not raw_machines:
            print(f"   ⚠️  [NO MACHINES] No equipment mentioned in methods.")
            cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = ?", (doi,))
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
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (doi, raw_machine, resolved_name, parent, ticker, conf))
            
            if resolved_name:
                status = f"→ {resolved_name} ({parent} / {ticker}) [conf: {conf:.2f}]"
                print(f"      ✅ {raw_machine} {status}")
            else:
                status = f"→ UNRESOLVED [conf: {conf:.2f}]"
                print(f"      ⚠️  {raw_machine} {status}")
        
        cursor.execute("UPDATE candidates SET processed = 1 WHERE doi = ?", (doi,))
        conn.commit()

    conn.close()
    print(f"\n{'='*70}")
    print("Pipeline execution complete.")
    print(f"{'='*70}")
# ==========================================
# EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    logger.info("Executing Semantic Pipeline Architecture...")
    # 1. Generate the registry first
    create_registry()
    
    # 2. Delete the old DB to apply the new schema
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        logger.info("Deleted old database to apply new schema.")
        
    # 3. Run pipeline
    pipeline_stage_1_and_2("2025-01-01", "2026-07-17")
    pipeline_stage_3_execution()