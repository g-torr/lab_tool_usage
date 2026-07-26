import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

client = instructor.from_openai(openrouter_client, mode=instructor.Mode.TOOLS)

class NoveltyClassification(BaseModel):
    reasoning: str = Field(description="Brief analysis: Is this in-house wet-lab data or public/computational?")
    is_novel_experiment: bool = Field(description="True if in-house wet-lab, False if public database/computational/review")

def classify_novelty(abstract: str, methods_snippet: str) -> NoveltyClassification:
    """
    Stage 1: Fast binary classification using only first 2000 chars of methods.
    This is cheap and filters out ~60% of papers immediately.
    """
    prompt = f"""
ABSTRACT:
{abstract}

METHODS SNIPPET (First 2000 chars):
{methods_snippet}

TASK: Determine if this paper generates novel in-house wet-lab experimental data.

CRITERIA:
- NOVEL (True): Authors performed experiments, generated new data, used physical instruments
- NOT NOVEL (False): Authors analyzed public databases (GEO, SRA), performed computational simulations, or wrote reviews

Respond with JSON only.
"""
    
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {"role": "system", "content": "You are a biomedical research classifier. Determine if a paper generates novel experimental data."},
            {"role": "user", "content": prompt}
        ],
        response_model=NoveltyClassification,
        max_retries=3
    )
    return response