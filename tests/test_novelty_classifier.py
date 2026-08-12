import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import novelty_classifier


class NoveltyClassifierTests(unittest.TestCase):
    def test_classify_novelty_prefers_local_zero_shot_classifier(self):
        local_result = novelty_classifier.NoveltyClassification(
            reasoning="Local zero-shot classifier judged this as novel wet-lab work.",
            is_novel_experiment=True,
        )

        with patch.object(novelty_classifier, "_classify_with_local_model", return_value=local_result), \
             patch.object(novelty_classifier, "_classify_with_remote_llm", side_effect=AssertionError("remote fallback should not be used")):
            result = novelty_classifier.classify_novelty("We isolated cells and generated new data.", "Methods details")

        self.assertTrue(result.is_novel_experiment)
        self.assertIn("local zero-shot", result.reasoning.lower())

    def test_classify_novelty_falls_back_to_remote_when_local_returns_none(self):
        remote_result = novelty_classifier.NoveltyClassification(
            reasoning="Remote LLM fallback decided this is not novel.",
            is_novel_experiment=False,
        )

        with patch.object(novelty_classifier, "_classify_with_local_model", return_value=None), \
             patch.object(novelty_classifier, "_classify_with_remote_llm", return_value=remote_result):
            result = novelty_classifier.classify_novelty("Data analysis of public datasets.", "Methods details")

        self.assertFalse(result.is_novel_experiment)
        self.assertIn("remote llm", result.reasoning.lower())

    def test_classify_novelty_skips_local_classifier_when_env_disables_it(self):
        remote_result = novelty_classifier.NoveltyClassification(
            reasoning="Remote LLM used because local classifier was disabled.",
            is_novel_experiment=False,
        )

        with patch.dict(os.environ, {"USE_LOCAL_STAGE1_CLASSIFIER": "false"}), \
             patch.object(novelty_classifier, "_classify_with_local_model", side_effect=AssertionError("local classifier should not be called")), \
             patch.object(novelty_classifier, "_classify_with_remote_llm", return_value=remote_result):
            result = novelty_classifier.classify_novelty("Data analysis of public datasets.", "Methods details")

        self.assertFalse(result.is_novel_experiment)
        self.assertIn("remote llm", result.reasoning.lower())


if __name__ == "__main__":
    unittest.main()
