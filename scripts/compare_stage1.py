import sys
import time
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForMaskedLM

# Force unbuffered prints so terminal updates instantly
sys.stdout.reconfigure(line_buffering=True)

print("="*80)
print("INITIALIZING STAGE 1 BENCHMARK SUITE")
print("="*80)

# -----------------------------------------------------------------------------
# 1. TEST DATASET
# -----------------------------------------------------------------------------
TEST_PAPERS = [
    {
        "doi": "10.1101/wet_lab_example",
        "expected": "NOVEL (True)",
        "expected_bool": True,
        "text": "Abstract: We investigated neural mechanisms in mice using two-photon optical imaging on a Leica SP8 confocal microscope and performed patch-clamp recordings."
    },
    {
        "doi": "10.1101/dry_lab_example",
        "expected": "PUBLIC/COMPUTATIONAL (False)",
        "expected_bool": False,
        "text": "Abstract: We conducted a meta-analysis. Fastq files were downloaded from NCBI GEO (GSE134355) and analyzed in R using Seurat."
    },
    {
        "doi": "10.1101/single_cell_reanalysis",
        "expected": "PUBLIC/COMPUTATIONAL (False)",
        "expected_bool": False,
        "text": "Single-cell multiome datasets (N=13 donors, 69,249 cells) were retrieved from GSE194122. Mutual information analysis was performed using custom R packages."
    },
    {
        "doi": "10.1101/in_vivo_trial",
        "expected": "NOVEL (True)",
        "expected_bool": True,
        "text": "We isolated primary human liver tissue and performed single-cell transcriptomics alongside immunohistochemistry staining to map tissue fibrosis."
    }
]

# Labels for NLI
NLI_LABELS = [
    "novel in-house physical wet-lab experiment",
    "computational re-analysis of public database or literature review"
]

# -----------------------------------------------------------------------------
# 2. MODEL INITIALIZATION
# -----------------------------------------------------------------------------

# Method 1: DeBERTa Zero-Shot Pipeline
print("[1/3] Loading DeBERTa-v3 NLI Zero-Shot Model...", flush=True)
nli_pipe = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", device=-1)

# Method 2: PubMedBERT Embedding Feature Extractor
print("[2/3] Loading PubMedBERT Embedding Model...", flush=True)
PUBMEDBERT_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
emb_tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_NAME)
emb_model = AutoModel.from_pretrained(PUBMEDBERT_NAME)
emb_model.eval()

# Method 3: PubMedBERT Masked LM (Fill-Mask)
print("[3/3] Loading PubMedBERT Masked LM Model...", flush=True)
mlm_model = AutoModelForMaskedLM.from_pretrained(PUBMEDBERT_NAME)
mlm_model.eval()

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_embedding(text: str):
    """Generates mean-pooled text embeddings."""
    inputs = emb_tokenizer(text[:1000], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Pre-compute anchor embeddings for Method 2
anchor_novel = get_embedding("In-house physical laboratory experiment generating new wet-lab biological data.")
anchor_public = get_embedding("Computational re-analysis of public databases GEO SRA datasets or literature review.")

def evaluate_mlm_masked(text: str):
    """
    Evaluates paper text using a cloze-style masked prompt in PubMedBERT.
    Prompt template: "<text> In summary, this study describes a [MASK] experiment."
    Compares the logit probabilities of token 'wet' vs 'computational'.
    """
    prompt = text[:1000] + " In summary, this study describes a [MASK] experiment."
    inputs = emb_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Locate [MASK] token index
    mask_token_id = emb_tokenizer.mask_token_id
    mask_pos = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1].item()
    
    with torch.no_grad():
        outputs = mlm_model(**inputs)
        logits = outputs.logits[0, mask_pos]
        probs = F.softmax(logits, dim=-1)
        
    token_wet_id = emb_tokenizer.convert_tokens_to_ids("wet")
    token_comp_id = emb_tokenizer.convert_tokens_to_ids("computational")
    
    prob_wet = probs[token_wet_id].item()
    prob_comp = probs[token_comp_id].item()
    
    # Normalize probabilities between the two target tokens
    denom = prob_wet + prob_comp + 1e-9
    norm_wet = prob_wet / denom
    norm_comp = prob_comp / denom
    
    pred_is_novel = norm_wet > norm_comp
    confidence = max(norm_wet, norm_comp)
    margin = abs(norm_wet - norm_comp)
    
    return pred_is_novel, confidence, margin, norm_wet, norm_comp

# -----------------------------------------------------------------------------
# 4. BENCHMARK EXECUTION
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("RUNNING METHOD COMPARISON EVALUATION")
print("="*80)

for i, paper in enumerate(TEST_PAPERS, 1):
    print(f"\n--- [Paper {i}] Expected: {paper['expected']} ---")
    print(f"DOI: {paper['doi']}")
    
    # -------------------------------------------------------------
    # Method 1: NLI Zero-Shot
    # -------------------------------------------------------------
    t0 = time.time()
    nli_res = nli_pipe(paper['text'], NLI_LABELS, hypothesis_template="This biomedical paper describes a {}.")
    nli_time = (time.time() - t0) * 1000
    nli_pred = nli_res['labels'][0] == NLI_LABELS[0]
    nli_conf = nli_res['scores'][0]
    
    # -------------------------------------------------------------
    # Method 2: Embedding Cosine Similarity
    # -------------------------------------------------------------
    t0 = time.time()
    text_emb = get_embedding(paper['text'])
    sim_novel = F.cosine_similarity(text_emb, anchor_novel).item()
    sim_public = F.cosine_similarity(text_emb, anchor_public).item()
    emb_time = (time.time() - t0) * 1000
    emb_pred = sim_novel > sim_public
    emb_margin = abs(sim_novel - sim_public)
    
    # -------------------------------------------------------------
    # Method 3: Masked LM (Fill-Mask Prompting)
    # -------------------------------------------------------------
    t0 = time.time()
    mlm_pred, mlm_conf, mlm_margin, p_wet, p_comp = evaluate_mlm_masked(paper['text'])
    mlm_time = (time.time() - t0) * 1000
    
    # -------------------------------------------------------------
    # Output Display
    # -------------------------------------------------------------
    print(f"  [1] NLI Zero-Shot (DeBERTa):")
    print(f"      • Pred: {'NOVEL' if nli_pred else 'PUBLIC'} | Conf: {nli_conf:.4f} | Latency: {nli_time:.1f}ms")
    
    print(f"  [2] Embedding Sim (PubMedBERT):")
    print(f"      • Pred: {'NOVEL' if emb_pred else 'PUBLIC'} | Margin: {emb_margin:.4f} (Wet:{sim_novel:.3f} vs Pub:{sim_public:.3f}) | Latency: {emb_time:.1f}ms")
    
    print(f"  [3] Masked MLM (PubMedBERT):")
    print(f"      • Pred: {'NOVEL' if mlm_pred else 'PUBLIC'} | Margin: {mlm_margin:.4f} (p_wet:{p_wet:.3f} vs p_comp:{p_comp:.3f}) | Latency: {mlm_time:.1f}ms")

print("\n" + "="*80)
print("BENCHMARK COMPLETED SUCCESSFULLY")
print("="*80)