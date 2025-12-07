from trl import SFTTrainer
from transformers import TrainingArguments
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class MinistralTrainer:
    def __init__(self, model: Any, tokenizer: Any, config: 'TrainingConfig'):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = None
    
    def setup_trainer(self, dataset: Any) -> SFTTrainer:
        """Setup the SFT trainer with given configuration"""
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=True,
            bf16=False,
            logging_steps=10,
            output_dir=self.config.output_dir,
            optim="adamw_8bit",
            seed=3407,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
        )
        
        return self.trainer
    
    def train(self) -> None:
        """Execute the training process"""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")
        
        logger.info(f"Starting training for {self.config.max_steps} steps...")
        self.trainer.train()
        logger.info("Training completed!")
