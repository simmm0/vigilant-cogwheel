#!/usr/bin/env python3
import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table

from ministral_finetuner.config import ConfigManager, TrainingConfig
from ministral_finetuner.dataset import load_dataset_with_quality
from ministral_finetuner.model import MinistralModel
from ministral_finetuner.trainer import MinistralTrainer

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Ministral-3B locally")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--output-dir", type=str, default="ministral-3b-finetuned")
    parser.add_argument("--model-name", type=str, help="Base model name (e.g., mistralai/Ministral-3B-Instruct-2410 or mistralai/Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--enable-annealing", action="store_true", help="Enable self-annealing dataset pipeline")
    parser.add_argument("--anneal-cycles", type=int, help="Number of annealing cycles")
    parser.add_argument("--anneal-initial-temp", type=float, help="Initial annealing temperature")
    parser.add_argument("--anneal-min-temp", type=float, help="Minimum annealing temperature")
    parser.add_argument("--anneal-cooling-rate", type=float, help="Cooling rate per cycle")
    parser.add_argument("--anneal-heating-rate", type=float, help="Heating rate per cycle")
    parser.add_argument("--anneal-stabilization-steps", type=int, help="Stabilization steps per cycle")
    parser.add_argument("--quality-min-length", type=int, help="Minimum token length for samples")
    parser.add_argument("--quality-min-coherence", type=float, help="Minimum coherence score")
    parser.add_argument("--quality-min-diversity", type=float, help="Minimum diversity score")
    parser.add_argument("--quality-max-repetition", type=float, help="Maximum repetition ratio")
    parser.add_argument("--bias-terms", nargs="*", help="Custom bias terms to flag during scoring")
    parser.add_argument("--cache-dir", type=str, help="Local cache directory for annealing artifacts")
    parser.add_argument("--dataset-cache-dir", type=str, help="Local cache directory for dataset artifacts")
    parser.add_argument("--disable-bias-check", action="store_true", help="Skip bias flagging during scoring")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = ConfigManager.load_from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Override with command line args
    config.dataset_path = args.dataset
    config.output_dir = args.output_dir
    config.local_cache_dir = args.dataset_cache_dir or config.local_cache_dir
    if args.model_name:
        config.model_name = args.model_name

    anneal_cfg = config.annealing
    if args.enable_annealing:
        anneal_cfg.enabled = True
    if args.anneal_cycles is not None:
        anneal_cfg.cycles = args.anneal_cycles
    if args.anneal_initial_temp is not None:
        anneal_cfg.initial_temperature = args.anneal_initial_temp
    if args.anneal_min_temp is not None:
        anneal_cfg.min_temperature = args.anneal_min_temp
    if args.anneal_cooling_rate is not None:
        anneal_cfg.cooling_rate = args.anneal_cooling_rate
    if args.anneal_heating_rate is not None:
        anneal_cfg.heating_rate = args.anneal_heating_rate
    if args.anneal_stabilization_steps is not None:
        anneal_cfg.stabilization_steps = args.anneal_stabilization_steps
    if args.quality_min_length is not None:
        anneal_cfg.quality_thresholds.min_length = args.quality_min_length
    if args.quality_min_coherence is not None:
        anneal_cfg.quality_thresholds.min_coherence = args.quality_min_coherence
    if args.quality_min_diversity is not None:
        anneal_cfg.quality_thresholds.min_diversity = args.quality_min_diversity
    if args.quality_max_repetition is not None:
        anneal_cfg.quality_thresholds.max_repetition = args.quality_max_repetition
    if args.bias_terms:
        anneal_cfg.bias_terms = args.bias_terms
    if args.cache_dir:
        anneal_cfg.cache.cache_dir = args.cache_dir
    if args.dataset_cache_dir:
        anneal_cfg.cache.dataset_cache_dir = args.dataset_cache_dir
    if args.disable_bias_check:
        anneal_cfg.bias_check_enabled = False

    try:
        # Load, score, and optionally anneal dataset
        console.print("[bold blue]Loading dataset...[/bold blue]")
        dataset, anneal_history = load_dataset_with_quality(config.dataset_path, config)
        if anneal_cfg.enabled:
            table = Table(title="Annealing Cycles")
            table.add_column("Cycle", justify="right")
            table.add_column("Temperature")
            table.add_column("Avg Quality")
            table.add_column("Size")
            for stat in anneal_history:
                table.add_row(
                    str(stat.cycle),
                    f"{stat.temperature:.3f}",
                    f"{stat.avg_quality:.3f}",
                    str(stat.size),
                )
            console.print(table)
        console.print(f"[green]Dataset ready with {len(dataset)} samples[/green]")

        # Setup model
        model_manager = MinistralModel(config.model_name)
        model, tokenizer = model_manager.load_model()
        model = model_manager.add_lora_adapters(r=config.lora_rank)

        # Train
        trainer = MinistralTrainer(model, tokenizer, config)
        trainer.setup_trainer(dataset)
        trainer.train()

        # Save
        model_manager.save_model(f"{config.output_dir}-final")
        if config.save_gguf:
            model_manager.save_gguf(f"{config.output_dir}-gguf", config.quantization_method)

        console.print(f"[bold green]Training complete! Models saved to {config.output_dir}[/bold green]")

    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
