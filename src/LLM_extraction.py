import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logger = logging.getLogger(__name__)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

client = instructor.from_openai(openrouter_client, mode=instructor.Mode.TOOLS)

class MachineExtraction(BaseModel):
    machines: List[str] = Field(description="List of physical hardware instruments mentioned. Empty list if none found.")

def extract_machines_from_chunk(methods_chunk: str) -> MachineExtraction:
    """
    Stage 2: Extract machines from a single chunk of text.
    Focused prompt with no classification logic.
    """
    if not methods_chunk or not methods_chunk.strip():
        return MachineExtraction(machines=[])

    prompt = f"""
METHODS TEXT CHUNK:
{methods_chunk}

TASK: Extract ALL physical hardware instruments, analyzers, and specialized equipment mentioned.

INCLUDE:
- Sequencers (Illumina, PacBio, Oxford Nanopore, etc.)
- Flow cytometers (BD FACSAria, Cytek Aurora, etc.)
- Mass spectrometers (Thermo Orbitrap, Bruker timsTOF, etc.)
- Microscopes (Zeiss LSM, Leica SP8, etc.)
- Electrophysiology rigs (Blackrock, Tucker-Davis, etc.)
- Any other specialized laboratory equipment

EXCLUDE:
- Reagents, chemicals, antibodies, biologicals
- Software, algorithms, computational tools
- General techniques (PCR, sequencing, etc.)
- Basic labware (centrifuges, syringes, tubes)

Respond with JSON only.
"""

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=[
                {"role": "system", "content": "You are a biomedical equipment extraction specialist. Identify all physical hardware instruments."},
                {"role": "user", "content": prompt}
            ],
            response_model=MachineExtraction,
            max_retries=1
        )
        if response is None:
            logger.warning("Machine extraction returned no response for chunk; falling back to empty result.")
            return MachineExtraction(machines=[])
        return response
    except Exception as exc:
        logger.warning("Machine extraction failed for chunk: %s", exc)
        return MachineExtraction(machines=[])

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 1000) -> List[str]:
    """
    Split text into overlapping chunks to avoid cutting sentences.
    Default: 8000 chars (~2000 tokens) with 1000 char overlap.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap to preserve context
    
    return chunks

def extract_all_machines(methods_text: str) -> List[str]:
    """
    Extract machines from full methods text by chunking if necessary.
    Deduplicates results across chunks.
    """
    if not methods_text or not methods_text.strip():
        return []

    chunks = chunk_text(methods_text, chunk_size=8000, overlap=1000)

    all_machines = []
    for i, chunk in enumerate(chunks):
        print(f"   Extracting machines from chunk {i+1}/{len(chunks)}...")
        try:
            result = extract_machines_from_chunk(chunk)
            machines = getattr(result, "machines", []) or []
            if isinstance(machines, list):
                all_machines.extend(machines)
        except Exception as exc:
            logger.warning("Skipping machine extraction chunk due to error: %s", exc)

    # Deduplicate (case-insensitive)
    seen = set()
    unique_machines = []
    for machine in all_machines:
        machine_lower = machine.lower()
        if machine_lower not in seen:
            seen.add(machine_lower)
            unique_machines.append(machine)

    return unique_machines