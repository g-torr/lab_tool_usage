import logging
import os
from typing import Optional

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

client = instructor.from_openai(openrouter_client, mode=instructor.Mode.TOOLS)


class NoveltyClassification(BaseModel):
    reasoning: str = Field(description="Brief analysis: Is this in-house wet-lab data or public/computational?")
    is_novel_experiment: bool = Field(description="True if in-house wet-lab, False if public database/computational/review")


class LocalStage1Classifier:
    """Tiny zero-shot classifier for stage-1 novelty filtering using Hugging Face."""

    def __init__(
        self,
        model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._classifier = None
        self.labels = [
            "novel wet-lab experiment generating original primary biological data",
            "purely computational study, literature review, or re-analysis of public database datasets with no lab experiment performed",
        ]
        self.hypothesis_template = "This biomedical paper describes a {}."

    def _load_pipeline(self):
        if self._classifier is not None:
            return self._classifier

        try:
            from transformers import pipeline
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised when transformers is unavailable
            raise RuntimeError("transformers/torch are required for local zero-shot classification") from exc

        device_id = self.device
        if device_id is None:
            device_id = 0 if torch.cuda.is_available() else -1

        self._classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device_id,
        )
        return self._classifier

    def classify_text(self, text: str, threshold: float = 0.50) -> dict:
        truncated_text = text[:1500]
        classifier = self._load_pipeline()
        result = classifier(
            truncated_text,
            self.labels,
            hypothesis_template=self.hypothesis_template,
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        is_novel = (top_label == self.labels[0]) and (top_score >= threshold)

        return {
            "is_novel": is_novel,
            "confidence": top_score,
            "top_label": top_label,
            "decision": "PROCEED_TO_STAGE_2" if is_novel else "FILTER_OUT",
        }


_local_stage1_classifier: Optional[LocalStage1Classifier] = None


def _get_local_stage1_classifier() -> Optional[LocalStage1Classifier]:
    global _local_stage1_classifier
    if _local_stage1_classifier is None:
        try:
            _local_stage1_classifier = LocalStage1Classifier()
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            logger.warning("Could not initialize local stage-1 classifier: %s", exc)
            return None
    return _local_stage1_classifier


def _classify_with_local_model(abstract: str, methods_snippet: str, threshold: float = 0.50) -> Optional[NoveltyClassification]:
    classifier = _get_local_stage1_classifier()
    if classifier is None:
        return None

    try:
        text = f"{abstract}\n\nMETHODS:\n{methods_snippet}"
        result = classifier.classify_text(text, threshold=threshold)
        return NoveltyClassification(
            reasoning=(
                f"Local zero-shot classifier labeled this as {result['top_label']} "
                f"with confidence {result['confidence']:.2f}."
            ),
            is_novel_experiment=result["is_novel"],
        )
    except Exception as exc:
        logger.warning("Local stage-1 classification failed: %s", exc)
        return None


def _classify_with_remote_llm(abstract: str, methods_snippet: str) -> NoveltyClassification:
    if not abstract or not methods_snippet:
        return NoveltyClassification(reasoning="Missing abstract or methods snippet.", is_novel_experiment=False)

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

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=[
                {"role": "system", "content": "You are a biomedical research classifier. Determine if a paper generates novel experimental data."},
                {"role": "user", "content": prompt},
            ],
            response_model=NoveltyClassification,
            max_retries=1,
        )
        if response is None:
            logger.warning("Novelty classification returned no response; defaulting to non-novel.")
            return NoveltyClassification(reasoning="LLM returned no usable response.", is_novel_experiment=False)
        return response
    except Exception as exc:
        logger.warning("Novelty classification failed: %s", exc)
        return NoveltyClassification(reasoning=f"LLM call failed: {exc}", is_novel_experiment=False)


def classify_novelty(abstract: str, methods_snippet: str) -> NoveltyClassification:
    """
    Stage 1: Fast binary classification using a local zero-shot model first,
    with the existing remote LLM as a fallback when the local model is unavailable.
    """
    if not abstract or not methods_snippet:
        return NoveltyClassification(reasoning="Missing abstract or methods snippet.", is_novel_experiment=False)

    use_local = os.getenv("USE_LOCAL_STAGE1_CLASSIFIER", "true").lower() not in {"0", "false", "no"}
    if use_local:
        local_result = _classify_with_local_model(abstract, methods_snippet)
        if local_result is not None:
            return local_result

    logger.info("Falling back to remote LLM classifier for novelty classification.")
    return _classify_with_remote_llm(abstract, methods_snippet)