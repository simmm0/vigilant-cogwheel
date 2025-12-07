from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from typing import Tuple, Optional

class MinistralModel:
    def __init__(self, model_name: str, load_in_4bit: bool = True):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> Tuple[Any, Any]:
        """Load the base model and tokenizer"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.model_name,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        return self.model, self.tokenizer
    
    def add_lora_adapters(self, r: int = 128, lora_alpha: int = 32, 
                         lora_dropout: float = 0.0) -> Any:
        """Add LoRA adapters to the model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return self.model
    
    def save_model(self, output_path: str, save_method: str = "merged_16bit"):
        """Save the fine-tuned model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded.")
            
        self.model.save_pretrained_merged(
            output_path, 
            self.tokenizer, 
            save_method=save_method
        )
    
    def save_gguf(self, output_path: str, quantization_method: str = "q5_k_m"):
        """Save model in GGUF format"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded.")
            
        self.model.save_pretrained_gguf(
            output_path, 
            self.tokenizer, 
            quantization_method=quantization_method
        )
