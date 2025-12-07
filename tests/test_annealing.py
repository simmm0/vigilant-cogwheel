from datasets import Dataset

from ministral_finetuner.annealing import AnnealingLoop
from ministral_finetuner.config import AnnealingSettings, QualityThresholds


def test_annealing_loop_runs_and_filters():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I am good, thank you for asking."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "spam spam spam spam"},
                    {"role": "assistant", "content": "spam spam spam spam"},
                ]
            },
        ]
    )

    anneal_cfg = AnnealingSettings(
        enabled=True,
        cycles=2,
        stabilization_steps=1,
        initial_temperature=1.2,
        min_temperature=0.5,
        convergence_delta=1.0,
        max_no_improve=1,
    )
    thresholds = QualityThresholds(min_length=3, min_coherence=0.1, min_diversity=0.1, max_repetition=0.5)
    loop = AnnealingLoop(anneal_cfg, thresholds)

    annealed, history = loop.run(dataset)

    assert len(history) >= 1
    assert len(annealed) <= len(dataset)
    assert "quality_score" in annealed.column_names

