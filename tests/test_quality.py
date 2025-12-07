from datasets import Dataset

from ministral_finetuner.config import QualityThresholds
from ministral_finetuner.dataset import QualityScorer


def _build_dataset(messages):
    return Dataset.from_list([{"messages": messages}])


def test_quality_scorer_accepts_balanced_sample():
    messages = [
        {"role": "user", "content": "Hello, can you explain transformers?"},
        {"role": "assistant", "content": "Sure, transformers are neural network architectures."},
    ]
    thresholds = QualityThresholds(min_length=2, min_coherence=0.0, min_diversity=0.1)
    scorer = QualityScorer(thresholds)

    result = scorer.score(messages)
    assert result["passes"] is True
    assert result["quality_score"] > 0
    assert result["flags"] == []


def test_quality_scorer_flags_repetition_and_length():
    messages = [
        {"role": "user", "content": "spam spam spam"},
        {"role": "assistant", "content": "spam spam spam"},
    ]
    thresholds = QualityThresholds(min_length=5, max_repetition=0.3, min_diversity=0.2)
    scorer = QualityScorer(thresholds)

    result = scorer.score(messages)
    assert result["passes"] is False
    assert "repetition" in result["flags"]
    assert "diversity" in result["flags"]

