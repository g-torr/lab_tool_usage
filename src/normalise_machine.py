import sqlite3


DB_NAME = "../db/local_market_share.db"

# Canonical Mapping: Map the noisy RAW_LLM strings to standardized market intelligence buckets
CANONICAL_MAPPING = {
    # Illumina Sequencers
    "Illumina_NovaSeq_6000": ["novaseq 6000", "novaseq6000", "illumina novaseq 6000"],
    "Illumina_NovaSeq_X": ["novaseq x", "novaseqx", "illumina novaseqx", "novaseq x plus"],
    "Illumina_NextSeq": ["nextseq", "next seq", "illumina nextseq"],
    "Illumina_MiSeq": ["miseq", "illumina miseq"],
    
    # PCR / qPCR
    "Standard_PCR": ["pcr machine", "thermal cycler", "pcr equipment"],
    "RealTime_PCR": ["rt-pcr", "rtqpcr", "real-time pcr", "real time pcr", "qpcr", "qrt-pcr"],
    
    # Microscopy
    "Confocal_Microscopy": ["confocal microscope", "confocal microscopy", "lsm 700", "lsm 800", "lsm 900", "lsm 980", "zeiss confocal"],
    "Two_Photon_Microscopy": ["two-photon", "two photon", "2-photon", "multiphoton"],
    
    # Nanopore
    "ONT_MinION": ["minion", "minion mk1b"],
    "ONT_PromethION": ["promethion", "promethion sequencer"],
    "ONT_GridION": ["gridion"],
    
    # PacBio
    "PacBio_Revio": ["pacbio revio", "revio system"],
    "PacBio_Sequel_IIe": ["sequel ii", "sequel iie", "sequel ii system"],
    
    # MRI / fMRI
    "MRI_Scanner": ["mri machine", "mri scanner", "fmri", "fmri machine", "fmri scanner", "3t mri", "7t mri"]
}

def normalize_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Fetch all RAW_LLM entries
    cursor.execute("SELECT id, standardized_machine FROM machine_mentions WHERE standardized_machine LIKE 'RAW_LLM: %'")
    rows = cursor.fetchall()

    updates = []
    for row_id, raw_name in rows:
        clean_name = raw_name.replace("RAW_LLM: ", "").lower().strip()
        matched_canonical = None
        
        # Check if it matches any of our canonical buckets
        for canonical, keywords in CANONICAL_MAPPING.items():
            if any(kw in clean_name for kw in keywords):
                matched_canonical = canonical
                break
                
        if matched_canonical:
            updates.append((matched_canonical, row_id))
        else:
            # Optional: If it's clearly a reagent/assay/software, delete it or flag it
            noise_keywords = ["fbs", "matrigel", "western blot", "crispr", "imagej", "matlab", "mice", "medium", "buffer", "kit", "reagent", "antibody"]
            if any(nk in clean_name for nk in noise_keywords):
                updates.append(("NON_MACHINE_NOISE", row_id))

    # Apply updates
    cursor.executemany("UPDATE machine_mentions SET standardized_machine = ? WHERE id = ?", updates)
    conn.commit()
    
    # Clean up the noise entirely
    cursor.execute("DELETE FROM machine_mentions WHERE standardized_machine = 'NON_MACHINE_NOISE'")
    conn.commit()
    
    print(f"Normalized {len(updates)} entries. Removed non-machine noise.")
    conn.close()

if __name__ == "__main__":
    normalize_database()