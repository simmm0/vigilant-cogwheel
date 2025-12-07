from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml

@dataclass
class QualityThresholds:
    min_length: int = 32
    max_length: Optional[int] = None
    min_user_ratio: float = 0.1
    max_user_ratio: float = 0.9
    min_coherence: float = 0.4
    min_diversity: float = 0.3
    max_repetition: float = 0.4
    toxicity_threshold: float = 0.5
    bias_threshold: float = 0.4

@dataclass
class CacheConfig:
    enable_cache: bool = True
    cache_dir: str = "cache/annealing"
    dataset_cache_dir: str = "cache/datasets"
    max_records: int = 50000

@dataclass
class AnnealingSettings:
    enabled: bool = False
    cycles: int = 3
    initial_temperature: float = 1.5
    min_temperature: float = 0.2
    cooling_rate: float = 0.85
    heating_rate: float = 1.05
    schedule: str = "linear_warmup_exp_decay"
    stabilization_steps: int = 1
    convergence_delta: float = 0.01
    max_no_improve: int = 2
    augmentation_enabled: bool = True
    structural_checks: bool = True
    bias_check_enabled: bool = True
    bias_terms: Optional[List[str]] = None
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    cache: CacheConfig = field(default_factory=CacheConfig)

@dataclass
class TrainingConfig:
    model_name: str = "mistralai/Ministral-3B-Instruct-2410"
    dataset_path: str = "my_dataset.jsonl"
    output_dir: str = "ministral-3b-finetuned"
    max_steps: int = 500
    batch_size: int = 16
    learning_rate: float = 2e-4
    lora_rank: int = 128
    max_seq_length: int = 32768
    save_gguf: bool = True
    quantization_method: str = "q5_k_m"
    local_cache_dir: str = "cache/datasets"
    annealing: AnnealingSettings = field(default_factory=AnnealingSettings)

@dataclass
class InferenceConfig:
    model_path: str = "ministral-3b-final"
    max_new_tokens: int = 512
    temperature: float = 0.7

class ConfigManager:
    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def load_from_yaml(path: str) -> TrainingConfig:
        data = ConfigManager._load_yaml(path)

        annealing_data = data.pop("annealing", {}) or {}
        quality_data = annealing_data.pop("quality_thresholds", {}) or {}
        cache_data = annealing_data.pop("cache", {}) or {}

        config = TrainingConfig(**data)

        # align cache defaults with training cache dir when present
        if "dataset_cache_dir" not in cache_data:
            cache_data["dataset_cache_dir"] = config.local_cache_dir

        quality_thresholds = QualityThresholds(**quality_data)
        cache_config = CacheConfig(**cache_data)
        annealing_settings = AnnealingSettings(
            quality_thresholds=quality_thresholds,
            cache=cache_config,
            **annealing_data,
        )
        config.annealing = annealing_settings
        return config
