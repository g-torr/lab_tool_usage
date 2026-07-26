import pandas as pd
import json

def create_registry():
    registry_data = [
        # --- GENOMICS & SEQUENCING (High Razor-and-Blade) ---
        {"canonical_name": "NovaSeq X / 6000", "vendor": "Illumina", "parent_company": "Illumina", "ticker": "ILMN", "category": "NGS Sequencing", "revenue_model": "Razor-Blade", "aliases": ["novaseq", "illumina novaseq", "novaseq x", "novaseq 6000", "hiseq", "solexa", "illumina short-read"]},
        {"canonical_name": "Revio / Sequel IIe", "vendor": "Pacific Biosciences", "parent_company": "Pacific Biosciences", "ticker": "PACB", "category": "Long-Read Sequencing", "revenue_model": "Razor-Blade", "aliases": ["pacbio", "revio", "sequel", "hifi", "pacbio hifi", "long-read sequencing"]},
        {"canonical_name": "PromethION / P2 Solo", "vendor": "Oxford Nanopore", "parent_company": "Oxford Nanopore", "ticker": "ONT", "category": "Long-Read Sequencing", "revenue_model": "Razor-Blade", "aliases": ["nanopore", "ont", "promethion", "minion", "p2solo", "p2 solo", "nanopore sequencing"]},
        {"canonical_name": "DNBSEQ-T7", "vendor": "MGI", "parent_company": "MGI Tech", "ticker": "Private (China)", "category": "NGS Sequencing", "revenue_model": "Razor-Blade", "aliases": ["mgi", "dnbseq", "bgi", "stereoseq"]},
        {"canonical_name": "AVITI", "vendor": "Element Biosciences", "parent_company": "Element Biosciences", "ticker": "Private", "category": "NGS Sequencing", "revenue_model": "Razor-Blade", "aliases": ["element", "aviti", "element biosciences"]},

        # --- SINGLE-CELL & SPATIAL OMICS (Premium Margins) ---
        {"canonical_name": "Chromium Controller", "vendor": "10x Genomics", "parent_company": "10x Genomics", "ticker": "TXG", "category": "Single-Cell", "revenue_model": "Razor-Blade", "aliases": ["10x", "chromium", "10x genomics", "chromium controller", "single-cell rnaseq", "scrna-seq"]},
        {"canonical_name": "Visium / Xenium", "vendor": "10x Genomics", "parent_company": "10x Genomics", "ticker": "TXG", "category": "Spatial Omics", "revenue_model": "Razor-Blade", "aliases": ["visium", "xenium", "10x visium", "10x xenium", "spatial transcriptomics"]},
        {"canonical_name": "MERSCOPE", "vendor": "Vizgen", "parent_company": "Vizgen", "ticker": "Private", "category": "Spatial Omics", "revenue_model": "Razor-Blade", "aliases": ["merscope", "merfish", "vizgen"]},
        {"canonical_name": "PhenoCycler", "vendor": "Akoya Biosciences", "parent_company": "Akoya Biosciences", "ticker": "AKYA", "category": "Spatial Omics", "revenue_model": "Razor-Blade", "aliases": ["phenocycler", "codex", "opal", "akoya"]},
        {"canonical_name": "CosMx / GeoMx", "vendor": "Standard BioTools", "parent_company": "Standard BioTools", "ticker": "LAB", "category": "Spatial Omics", "revenue_model": "Razor-Blade", "aliases": ["cosmx", "geomx", "nanostring", "standard biotools"]},

        # --- FLOW CYTOMETRY & CELL SORTING ---
        {"canonical_name": "FACSAria / LSRFortessa", "vendor": "BD Biosciences", "parent_company": "BD", "ticker": "BDX", "category": "Flow Cytometry", "revenue_model": "Razor-Blade", "aliases": ["bd facs", "facsaria", "lsrfortessa", "fortessa", "bd rhapsody", "flow cytometer", "facs"]},
        {"canonical_name": "CytoFLEX", "vendor": "Beckman Coulter", "parent_company": "Danaher", "ticker": "DHR", "category": "Flow Cytometry", "revenue_model": "Razor-Blade", "aliases": ["cytoflex", "beckman coulter", "beckman"]},
        {"canonical_name": "Aurora", "vendor": "Cytek", "parent_company": "Cytek", "ticker": "CTKB", "category": "Flow Cytometry", "revenue_model": "Hybrid", "aliases": ["cytek", "aurora", "full spectrum"]},
        {"canonical_name": "ID7000", "vendor": "Sony Biotechnology", "parent_company": "Sony", "ticker": "Private (Sub)", "category": "Flow Cytometry", "revenue_model": "Razor-Blade", "aliases": ["sony", "id7000", "sony id7000", "spectral cell analyzer"]},

        # --- MASS SPECTROMETRY & PROTEOMICS (Expanded for Imaging) ---
        {"canonical_name": "Orbitrap / Q Exactive", "vendor": "Thermo Fisher", "parent_company": "Thermo Fisher", "ticker": "TMO", "category": "Mass Spectrometry", "revenue_model": "Razor-Blade", "aliases": ["orbitrap", "thermo", "q exactive", "fusion lumos", "thermo fisher", "mass spectrometry", "lc-ms"]},
        {"canonical_name": "timsTOF / timsTOF fleX", "vendor": "Bruker", "parent_company": "Bruker", "ticker": "BRKR", "category": "Mass Spectrometry & Imaging", "revenue_model": "Hardware/Service", "aliases": ["bruker", "timstof", "timstof flex", "maldi", "fticr", "maldi-2", "ultraflextreme", "rapiflex"]},
        {"canonical_name": "Shimadzu iMScope", "vendor": "Shimadzu", "parent_company": "Shimadzu", "ticker": "Private (Japan)", "category": "Imaging Mass Spectrometry", "revenue_model": "Razor-Blade", "aliases": ["imscope", "shimadzu", "imscope trio", "imscope qt", "maldi imaging"]},
        {"canonical_name": "Waters SYNAPT / MALDI", "vendor": "Waters", "parent_company": "Waters", "ticker": "WAT", "category": "Mass Spectrometry", "revenue_model": "Razor-Blade", "aliases": ["waters", "synapt", "g2-si", "hdms", "maldi synapt"]},
        {"canonical_name": "MIBI / Multiplexed Ion Beam Imaging", "vendor": "Standard BioTools", "parent_company": "Standard BioTools", "ticker": "LAB", "category": "Spatial Proteomics", "revenue_model": "Razor-Blade", "aliases": ["mibi", "multiplexed ion beam imaging", "fluidigm mibi", "standard biotools mibi"]}, 
        # --- ADVANCED IMAGING & MICROSCOPY ---
        {"canonical_name": "SP8 / Stellaris", "vendor": "Leica", "parent_company": "Danaher", "ticker": "DHR", "category": "Microscopy", "revenue_model": "Hardware", "aliases": ["leica", "sp8", "stellaris", "leica microsystems", "confocal"]},
        {"canonical_name": "LSM / Axio", "vendor": "Zeiss", "parent_company": "Carl Zeiss", "ticker": "Private", "category": "Microscopy", "revenue_model": "Hardware", "aliases": ["zeiss", "lsm", "axio", "confocal"]},
        {"canonical_name": "Ti-E / Eclipse", "vendor": "Nikon", "parent_company": "Nikon", "ticker": "Private", "category": "Microscopy", "revenue_model": "Hardware", "aliases": ["nikon", "ti-e", "eclipse", "nikon instruments"]},
    
        # --- MICROSCOPY SUB-COMPONENTS (High-Value Add-ons) ---
        {"canonical_name": "Mai Tai / Spectra-Physics", "vendor": "Spectra-Physics", "parent_company": "MKS Instruments", "ticker": "MKSI", "category": "Microscopy Laser", "revenue_model": "Hardware", "aliases": ["spectra-physics", "spectraphysics", "mai tai", "titanium sapphire laser", "ultrafast laser"]},
        {"canonical_name": "ThorLabs Optical Components", "vendor": "ThorLabs", "parent_company": "ThorLabs", "ticker": "Private", "category": "Optical Components", "revenue_model": "Hardware", "aliases": ["thorlabs", "thor labs", "detector", "photodiode"]},
        {"canonical_name": "Physik Instrumente (PI)", "vendor": "Physik Instrumente", "parent_company": "Physik Instrumente", "ticker": "Private", "category": "Precision Motion", "revenue_model": "Hardware", "aliases": ["physik instrumente", "piezo lens holder", "piezo stage", "pi"]},
        {"canonical_name": "Galvo / Resonance Scanners", "vendor": "Cambridge Tech / Scanlab", "parent_company": "Various", "ticker": "Various", "category": "Optical Scanning", "revenue_model": "Hardware", "aliases": ["galvo-mirror", "resonance scanner", "galvanometer scanner", "scan head"]},
        {"canonical_name": "Custom / Academic Rig", "vendor": "Academic / Custom", "parent_company": "N/A", "ticker": "N/A", "category": "Custom Microscopy", "revenue_model": "N/A", "aliases": ["custom built", "independent neuroscience services", "home-built", "custom microscope"]},

        # --- NEUROTECH & ELECTROPHYSIOLOGY ---
        {"canonical_name": "Cerebus / Utah Array", "vendor": "Blackrock Neurotech", "parent_company": "Blackrock", "ticker": "Private", "category": "Neurotech", "revenue_model": "Hardware", "aliases": ["blackrock", "cerebus", "utah array", "neuropixels", "multielectrode array"]},
        {"canonical_name": "EyeLink 1000", "vendor": "SR Research", "parent_company": "SR Research", "ticker": "Private", "category": "Eye Tracking", "revenue_model": "Hardware", "aliases": ["eyelink", "sr research", "eye-link", "eye tracker"]},
            # --- ADD THESE TO YOUR EXISTING registry_data LIST IN create_registry.py ---
    
    # QC & Electrophoresis (High consumable attach rate, Agilent $A)
    {"canonical_name": "TapeStation / ScreenTape", "vendor": "Agilent", "parent_company": "Agilent", "ticker": "A", "category": "Electrophoresis / QC", "revenue_model": "Razor-Blade", "aliases": ["tapestation", "screentape", "agilent tapestation", "genomic dna screentape"]},
    
    # Cell Separation (Massive market share, critical for scRNA-seq prep)
    {"canonical_name": "MACS Cell Separation System", "vendor": "Miltenyi Biotec", "parent_company": "Miltenyi Biotec", "ticker": "Private", "category": "Cell Separation", "revenue_model": "Razor-Blade", "aliases": ["macs", "miltenyi", "macs system", "macs separator"]},
    
    # Imaging & Blotting (Core lab staples)
    {"canonical_name": "GelDoc / ChemiDoc", "vendor": "Bio-Rad", "parent_company": "Bio-Rad", "ticker": "Private", "category": "Imaging", "revenue_model": "Hardware/Consumables", "aliases": ["geldoc", "chemidoc", "bio-rad geldoc", "imaging system"]},
    {"canonical_name": "Revolve Microscope", "vendor": "Echo", "parent_company": "Echo", "ticker": "Private", "category": "Microscopy", "revenue_model": "Hardware", "aliases": ["revolve", "echo revolve", "rvl2", "widefield microscope"]},
    
    # Sample Prep (High-frequency use)
    {"canonical_name": "Bioruptor", "vendor": "Diagenode", "parent_company": "Diagenode", "ticker": "Private", "category": "Sample Prep", "revenue_model": "Hardware", "aliases": ["bioruptor", "diagenode", "sonicator"]},
    
    # Proteomics / Mass Spec Adjacent (Thermo Fisher $TMO ecosystem)
    {"canonical_name": "Easy-nLC", "vendor": "Thermo Fisher", "parent_company": "Thermo Fisher", "ticker": "TMO", "category": "Chromatography", "revenue_model": "Razor-Blade", "aliases": ["easy-nlc", "nano lc", "hplc system"]},
    {"canonical_name": "FAIMS Pro", "vendor": "Thermo Fisher", "parent_company": "Thermo Fisher", "ticker": "TMO", "category": "Mass Spectrometry", "revenue_model": "Hardware Add-on", "aliases": ["faims", "faims pro", "faims pro interface"]}
    ]

    df_registry = pd.DataFrame(registry_data)
    # Format aliases as JSON strings for CSV storage
    df_registry['aliases'] = df_registry['aliases'].apply(json.dumps)
    df_registry.to_csv('machine_registry.csv', index=False)
    print(f"Registry created successfully with {len(df_registry)} high-value platforms.")

if __name__ == "__main__":
    create_registry()