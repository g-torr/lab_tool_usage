import pandas as pd
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util, CrossEncoder

class SemanticMachineResolver:
    def __init__(self, 
                 registry_path='machine_registry.csv', 
                 bi_encoder_model='all-MiniLM-L6-v2', 
                 cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        
        print(f"Loading registry from {registry_path}...")
        self.registry = pd.read_csv(registry_path)
        
        def build_search_text(row):
            aliases = json.loads(row['aliases']) if pd.notna(row['aliases']) else []
            return f"{row['canonical_name']} {' '.join(aliases)}".lower()
        
        self.registry['search_text'] = self.registry.apply(build_search_text, axis=1)
        
        print(f"Loading Bi-Encoder model ({bi_encoder_model})...")
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.embeddings = self.bi_encoder.encode(self.registry['search_text'].tolist(), convert_to_tensor=True)
        
        print(f"Loading Cross-Encoder model ({cross_encoder_model})...")
        # Cross-encoders evaluate query + candidate TOGETHER, solving contextual ambiguities
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        self.bi_threshold = 0.75      # Fast-track threshold for obvious matches
        self.cross_threshold = 4.0    # Typical strong match logit for ms-marco models

    def _clean_query(self, text: str) -> str:
        """Remove syntactic noise that causes embedding dilution."""
        # 1. Remove parentheticals (e.g., "(BD Biosciences)", "(Thermo Fisher)")
        cleaned = re.sub(r'\([^)]*\)', '', text)
        # 2. Remove generic filler words that dilute the core entity signal
        filler_words = ["platform", "system", "using", "kit", "reagent", "version", "software", "device"]
        for word in filler_words:
            cleaned = re.sub(rf'\b{word}\b', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def resolve(self, raw_llm_string: str) -> tuple:
        """
        Returns: (canonical_name, parent_company, ticker, confidence_score)
        """
        if not raw_llm_string or not isinstance(raw_llm_string, str):
            return None, None, None, 0.0
            
        cleaned_query = self._clean_query(raw_llm_string)
        
        # ==========================================
        # STAGE 1: Bi-Encoder Retrieval (Fast)
        # ==========================================
        query_embedding = self.bi_encoder.encode([cleaned_query.lower()], convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top 5 candidates
        top_k = min(5, len(cos_scores))
        top_results = cos_scores.topk(top_k)
        top_indices = top_results.indices.tolist()
        top_scores = top_results.values.tolist()
        
        # Fast-track: If the best match is overwhelmingly obvious, skip cross-encoder
        if top_scores[0] >= self.bi_threshold:
            best_idx = top_indices[0]
            row = self.registry.iloc[best_idx]
            return row['canonical_name'], row['parent_company'], row['ticker'], float(top_scores[0])

        # ==========================================
        # STAGE 2: Cross-Encoder Re-ranking (Context-Aware)
        # ==========================================
        # This evaluates the query and candidate TOGETHER, solving ambiguities 
        # like "Thermo Ultimate 3000" vs "Orbitrap"
        pairs = [(cleaned_query.lower(), self.registry.iloc[idx]['search_text']) for idx in top_indices]
        cross_scores = self.cross_encoder.predict(pairs)
        
        best_cross_idx = int(np.argmax(cross_scores))
        best_cross_score = float(cross_scores[best_cross_idx])
        best_registry_idx = top_indices[best_cross_idx]
        
        if best_cross_score >= self.cross_threshold:
            row = self.registry.iloc[best_registry_idx]
            # Normalize cross-encoder logit (typically -10 to +10) to a 0.50–0.99 confidence score
            normalized_conf = min(0.99, max(0.50, (best_cross_score + 5) / 10)) 
            return row['canonical_name'], row['parent_company'], row['ticker'], normalized_conf
            
        # Fallback: Return the best bi-encoder score as the confidence metric if both fail
        return None, None, None, float(top_scores[0])