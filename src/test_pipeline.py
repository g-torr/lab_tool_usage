"""
Pipeline Integration Test
Tests the complete flow: Methods Extraction → Novelty Classification → Machine Extraction → Semantic Resolution
"""
import os
import sys
import sqlite3
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import fetch_methods_text_from_web, init_db, DB_NAME
from novelty_classifier import classify_novelty
from machine_extractor import extract_machines_from_chunk_fast
from semantic_resolver import SemanticMachineResolver
from create_registry import create_registry

load_dotenv()

def test_full_pipeline():
    """
    Test the complete pipeline on a known good example from unmatched.csv
    """
    print("="*80)
    print("PIPELINE INTEGRATION TEST")
    print("="*80)
    
    # Test DOI from unmatched.csv that has known hardware mentions
    test_doi = "10.1101/2024.12.31.630917"
    mock_abstract = "We developed a new in vivo imaging technique to study neural circuits using advanced optical hardware."
    
    print(f"\n[1/4] Testing Methods Extraction for DOI: {test_doi}")
    print("-" * 80)
    
    # Step 1: Fetch methods text
    methods_text = fetch_methods_text_from_web(test_doi)
    if not methods_text:
        print("❌ FAILED: Could not extract methods text")
        return False
    
    print(f"✅ SUCCESS: Extracted {len(methods_text)} characters")
    print(f"   First 200 chars: {methods_text[:200]}...")
    
    # Step 2: Test Novelty Classification (Stage 1)
    print(f"\n[2/4] Testing Novelty Classification (First 2000 chars)")
    print("-" * 80)
    try:
        novelty_result = classify_novelty(mock_abstract, methods_text[:2000])
        print(f"   ✅ Novelty Result Received")
        print(f"      Is Novel: {novelty_result.is_novel_experiment}")
        print(f"      Reasoning: {novelty_result.reasoning}")
        
        if not novelty_result.is_novel_experiment:
            print("   ⚠️  Classifier marked as NON-NOVEL. Pipeline would stop here.")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 3: Test Machine Extraction (Stage 2 - Chunked)
    print(f"\n[3/4] Testing Machine Extraction (Chunked)")
    print("-" * 80)
    try:
        raw_machines = extract_machines_from_chunk_fast(methods_text)
        print(f"   ✅ Machine Extraction Received")
        print(f"      Total Unique Machines Found: {len(raw_machines)}")
        if raw_machines:
            print(f"      Sample Machines: {raw_machines[:5]}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

    # Step 4: Test Semantic Resolver (Stage 3)
    print(f"\n[4/4] Testing Semantic Resolver")
    print("-" * 80)
    try:
        # Create registry if it doesn't exist
        if not os.path.exists('machine_registry.csv'):
            print("   Creating machine registry...")
            create_registry()
        
        print("   Initializing SemanticMachineResolver...")
        resolver = SemanticMachineResolver('machine_registry.csv')
        print("   ✅ Resolver initialized successfully")
        
        if not raw_machines:
            print("   ⚠️  No machines extracted, skipping resolution test")
        else:
            print(f"\n   Testing resolution on {len(raw_machines)} machines...")
            for machine in raw_machines:
                resolved_name, parent, ticker, confidence = resolver.resolve(machine)
                if resolved_name:
                    print(f"   ✅ '{machine}' → {resolved_name} ({parent} / {ticker}) [conf: {confidence:.2f}]")
                else:
                    print(f"   ⚠️  '{machine}' → UNRESOLVED [conf: {confidence:.2f}]")
                    
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Methods Extraction: SUCCESS")
    print(f"✅ Novelty Classification: SUCCESS")
    print(f"✅ Machine Extraction: SUCCESS ({len(raw_machines)} machines)")
    print(f"✅ Semantic Resolver: SUCCESS")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)