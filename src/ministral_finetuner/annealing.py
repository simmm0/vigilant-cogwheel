from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Tuple

from datasets import Dataset

from ministral_finetuner.config import AnnealingSettings, QualityThresholds
from ministral_finetuner.dataset import QualityScorer


AnnealAugment = Callable[[Dataset, float], Dataset]


@dataclass
class AnnealingStats:
    cycle: int
    temperature: float
    avg_quality: float
    size: int
    flags_removed: Dict[str, int]


class AnnealingLoop:
    """Simple self-annealing loop with heating/stabilization/cooling."""

    def __init__(
        self,
        config: AnnealingSettings,
        thresholds: QualityThresholds,
        bias_terms: Optional[List[str]] = None,
        augment_fn: Optional[AnnealAugment] = None,
    ):
        self.config = config
        self.scorer = QualityScorer(
            thresholds,
            bias_terms=bias_terms,
            enable_bias=config.bias_check_enabled,
        )
        self.augment_fn = augment_fn

    def _temperature_filter(self, dataset: Dataset, temperature: float) -> Tuple[Dataset, Dict[str, int]]:
        """Filter samples based on quality while allowing temperature-based slack."""
        slack = max(0.0, temperature - 1.0) * 0.05
        thresholds = self.scorer.thresholds
        removed_flags: Dict[str, int] = {}

        def _accept(example: Dict[str, Any]) -> bool:
            metrics = example.get("quality_metrics", {})
            if not metrics:
                return True
            length_ok = metrics.get("length", 0) >= thresholds.min_length * (1 - slack)
            if thresholds.max_length:
                length_ok = length_ok and metrics.get("length", 0) <= thresholds.max_length * (1 + slack)
            user_ratio = metrics.get("user_ratio", 0)
            user_ok = thresholds.min_user_ratio - slack <= user_ratio <= thresholds.max_user_ratio + slack
            coherence_ok = metrics.get("coherence", 0) + slack >= thresholds.min_coherence
            diversity_ok = metrics.get("diversity", 0) + slack >= thresholds.min_diversity
            repetition_ok = metrics.get("repetition", 1) <= thresholds.max_repetition + slack
            toxicity_ok = metrics.get("toxicity", 0) <= thresholds.toxicity_threshold + slack
            bias_ok = metrics.get("bias", 0) <= thresholds.bias_threshold + slack
            accepted = all([length_ok, user_ok, coherence_ok, diversity_ok, repetition_ok, toxicity_ok, bias_ok])
            if not accepted:
                for flag in example.get("quality_flags", []):
                    removed_flags[flag] = removed_flags.get(flag, 0) + 1
            return accepted

        return dataset.filter(_accept), removed_flags

    @staticmethod
    def _avg_quality(dataset: Dataset) -> float:
        if len(dataset) == 0:
            return 0.0
        scores = dataset["quality_score"] if "quality_score" in dataset.column_names else []
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def run(self, dataset: Dataset) -> Tuple[Dataset, List[AnnealingStats]]:
        if not self.config.enabled:
            return dataset, []

        current = self.scorer.annotate_dataset(dataset)
        temperature = self.config.initial_temperature
        best_quality = self._avg_quality(current)
        stagnant_steps = 0
        history: List[AnnealingStats] = []

        for cycle in range(self.config.cycles):
            # Heating
            temperature *= self.config.heating_rate
            if self.augment_fn and self.config.augmentation_enabled:
                current = self.augment_fn(current, temperature)
                current = self.scorer.annotate_dataset(current)

            # Stabilization rounds
            for _ in range(max(1, self.config.stabilization_steps)):
                current = self.scorer.annotate_dataset(current)

            # Cooling and filtering
            temperature = max(self.config.min_temperature, temperature * self.config.cooling_rate)
            current, removed_flags = self._temperature_filter(current, temperature)
            current = self.scorer.annotate_dataset(current)

            avg_quality = self._avg_quality(current)
            history.append(
                AnnealingStats(
                    cycle=cycle,
                    temperature=temperature,
                    avg_quality=avg_quality,
                    size=len(current),
                    flags_removed=removed_flags,
                )
            )

            # Convergence checks
            if avg_quality - best_quality < self.config.convergence_delta:
                stagnant_steps += 1
            else:
                best_quality = avg_quality
                stagnant_steps = 0

            if stagnant_steps >= self.config.max_no_improve or temperature <= self.config.min_temperature:
                break

        return current, history

