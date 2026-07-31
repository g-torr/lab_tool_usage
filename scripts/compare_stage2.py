import time
import re
from gliner import GLiNER
import torch
import logging
logger = logging.getLogger(__name__)
# Determine device automatically
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Initializing GLiNER-2.1 Large model on device: {device}...")
# Keep the EXACT same model instance
print("Loading model urchade/gliner_large-v2.1...")
model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True).to(device)

# Refined labels: Specific instrument categories instead of broad generic buckets
REFINED_LABELS = [
    "dna sequencer",
    "rna sequencer",
    "microscope",
    "flow cytometer",
    "cell sorter",
    "mass spectrometer",
    "single-cell instrument",
    "chromatography system",
    "electrophysiology equipment"
]

# Consumables and components filter
NOISE_TERMS = {
    "plate", "plates", "well", "wells", "objective", "lens", 
    "buffer", "kit", "reagent", "tube", "tubes", "dish", "coverslip"
}

def post_filter_entities(entities: list) -> list:
    cleaned = []
    for ent in entities:
        text = ent['text'].strip()
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Drop if the span is purely/mainly a generic consumable or optical accessory
        if words.intersection(NOISE_TERMS):
            continue
            
        cleaned.append(text)
    return sorted(list(set(cleaned)))

# Same sample chunks from your previous run
SAMPLE_CHUNKS = [
    """
    Imaging was performed using a Leica SP8 confocal microscope equipped with a 63x oil-immersion objective.
    Single-cell RNA sequencing libraries were prepared with the 10x Genomics Chromium Next GEM Single Cell 3' Kit (v3.1)
    and sequenced on an Illumina NovaSeq 6000 platform to a depth of 50,000 reads per cell.
    """,
    """
    Cells were sorted on a BD FACSAria III cell sorter into 96-well plates. Protein identification was carried out on an
    Orbitrap Fusion Lumos Tribrid mass spectrometer coupled to an EASY-nLC 1200 system (Thermo Fisher Scientific).
    Electrophysiology recordings were gathered using a Axon Multiclamp 700B amplifier.
    """
]

print("\n" + "="*80)
print("RUNNING FORMATTING & FILTERING TEST ON GLiNER-2.1")
print("="*80)

for i, text in enumerate(SAMPLE_CHUNKS, 1):
    t0 = time.time()
    
    # 1. Run zero-shot with refined labels & higher confidence threshold
    raw_entities = model.predict_entities(
        text, 
        REFINED_LABELS, 
        threshold=0.45,
        flat_ner=True
    )
    
    latency = (time.time() - t0) * 1000
    
    # 2. Apply post-filter
    final_machines = post_filter_entities(raw_entities)
    
    print(f"\n--- [Chunk {i}] Latency: {latency:.1f}ms ---")
    print("Raw Predictions from Model:")
    for ent in raw_entities:
        print(f"  • '{ent['text']}' [{ent['label']}] (conf: {ent['score']:.4f})")
        
    print(f"\nFinal Filtered Output ({len(final_machines)} items):")
    print(f"  --> {final_machines}")

print("\n" + "="*80)