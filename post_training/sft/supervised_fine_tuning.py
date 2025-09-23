"""
Supervised Fine-Tuning (SFT) - Improved Version

This module provides functionality for supervised fine-tuning of language models
with improved error handling, logging, and code organization.
"""

import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter specific warnings only
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Configuration constants
class Config:
    """Configuration constants for the SFT pipeline"""
    DEFAULT_MAX_NEW_TOKENS = 100
    DEFAULT_LEARNING_RATE = 8e-5
    DEFAULT_EPOCHS = 1
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
    DEFAULT_LOGGING_STEPS = 2
    DATASET_PREVIEW_ROWS = 3
    SMALL_DATASET_SIZE = 100


class ModelManager:
    """Manages model and tokenizer loading, testing, and cleanup"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    @contextmanager
    def load_model_context(self, model_path: Union[str, Path]):
        """Context manager for safe model loading and cleanup"""
        model, tokenizer = None, None
        try:
            model, tokenizer = self._load_model_and_tokenizer(model_path)
            yield model, tokenizer
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
        finally:
            self._cleanup_model(model, tokenizer)
    
    def _load_model_and_tokenizer(self, model_name: Union[str, Path]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer with proper configuration"""
        try:
            logger.info(f"Loading model and tokenizer from {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Move model to appropriate device
            model.to(self.device)
            
            # Configure chat template if not present
            self._configure_chat_template(tokenizer)
            
            # Configure padding token
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _configure_chat_template(self, tokenizer: PreTrainedTokenizer) -> None:
        """Configure chat template if not present"""
        if not tokenizer.chat_template:
            tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""
            logger.info("Configured default chat template")
    
    def _cleanup_model(self, model: Optional[PreTrainedModel], tokenizer: Optional[PreTrainedTokenizer]) -> None:
        """Clean up model and tokenizer resources"""
        try:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            self._cleanup_memory()
            logger.info("Model cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def _cleanup_memory(self) -> None:
        """Clean up memory and CUDA cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelTester:
    """Handles model testing and response generation"""
    
    @staticmethod
    def generate_response(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        user_message: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = Config.DEFAULT_MAX_NEW_TOKENS
    ) -> str:
        """Generate response from model for given input"""
        try:
            # Format messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Extract and decode response
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    @staticmethod
    def test_model_with_questions(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        questions: List[str],
        system_message: Optional[str] = None,
        title: str = "Model Output"
    ) -> None:
        """Test model with a list of questions"""
        logger.info(f"Testing model: {title}")
        print(f"\n=== {title} ===")
        
        for i, question in enumerate(questions, 1):
            try:
                response = ModelTester.generate_response(
                    model, tokenizer, question, system_message
                )
                print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")
            except Exception as e:
                logger.error(f"Error testing question {i}: {e}")
                print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\nError: {e}\n")


class DatasetManager:
    """Manages dataset loading and display"""
    
    @staticmethod
    def load_training_dataset(dataset_name: str, limit_size: Optional[int] = None) -> Dataset:
        """Load training dataset with optional size limit"""
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)["train"]
            
            if limit_size:
                dataset = dataset.select(range(min(limit_size, len(dataset))))
                logger.info(f"Limited dataset to {len(dataset)} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    @staticmethod
    def display_dataset_preview(dataset: Dataset, num_rows: int = Config.DATASET_PREVIEW_ROWS) -> None:
        """Display a preview of the dataset"""
        try:
            logger.info("Displaying dataset preview")
            rows = []
            
            for i in range(min(num_rows, len(dataset))):
                example = dataset[i]
                
                # Extract user and assistant messages
                user_msg = next(
                    (m["content"] for m in example["messages"] if m["role"] == "user"),
                    "No user message found"
                )
                assistant_msg = next(
                    (m["content"] for m in example["messages"] if m["role"] == "assistant"),
                    "No assistant message found"
                )
                
                rows.append({
                    "User Prompt": user_msg,
                    "Assistant Response": assistant_msg
                })
            
            # Display as formatted table
            df = pd.DataFrame(rows)
            pd.set_option("display.max_colwidth", None)
            print("\n=== Dataset Preview ===")
            print(df.to_string(index=False))
            print()
            
        except Exception as e:
            logger.error(f"Error displaying dataset preview: {e}")


class SFTManager:
    """Manages the supervised fine-tuning process"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    def create_sft_config(
        self,
        learning_rate: float = Config.DEFAULT_LEARNING_RATE,
        num_epochs: int = Config.DEFAULT_EPOCHS,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps: int = Config.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        logging_steps: int = Config.DEFAULT_LOGGING_STEPS
    ) -> SFTConfig:
        """Create SFT configuration with memory-friendly settings"""
        return SFTConfig(
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=False,  # Disabled for memory efficiency
            logging_steps=logging_steps,
            dataloader_num_workers=0,  # Disable multiprocessing for stability
            dataloader_pin_memory=False,  # Reduce memory pressure
        )
    
    def train_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        config: Optional[SFTConfig] = None
    ) -> SFTTrainer:
        """Train model using supervised fine-tuning"""
        try:
            if config is None:
                config = self.create_sft_config()
            
            logger.info("Starting supervised fine-tuning")
            
            trainer = SFTTrainer(
                model=model,
                args=config,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            
            trainer.train()
            logger.info("Training completed successfully")
            
            return trainer
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main execution function"""
    try:
        # Configuration
        use_gpu = False
        test_questions = [
            "Give me an 1-sentence introduction of LLM.",
            "Calculate 1+1-1",
            "What's the difference between thread and process?",
        ]
        
        # Initialize managers
        model_manager = ModelManager(use_gpu=use_gpu)
        dataset_manager = DatasetManager()
        sft_manager = SFTManager(use_gpu=use_gpu)
        
        # Test base models
        base_model_path = "./models/Qwen/Qwen3-0.6B-Base"
        sft_model_path = "./models/banghua/Qwen3-0.6B-SFT"
        
        logger.info("Testing base model")
        with model_manager.load_model_context(base_model_path) as (model, tokenizer):
            ModelTester.test_model_with_questions(
                model, tokenizer, test_questions, title="Base Model (Before SFT) Output"
            )
        
        logger.info("Testing pre-trained SFT model")
        with model_manager.load_model_context(sft_model_path) as (model, tokenizer):
            ModelTester.test_model_with_questions(
                model, tokenizer, test_questions, title="Pre-trained SFT Model Output"
            )
        
        # Perform SFT on small model
        small_model_path = "./models/HuggingFaceTB/SmolLM2-135M"
        dataset_name = "banghua/DL-SFT-Dataset"
        
        logger.info("Loading small model for training")
        with model_manager.load_model_context(small_model_path) as (model, tokenizer):
            # Load and preview dataset
            dataset_limit = Config.SMALL_DATASET_SIZE if not use_gpu else None
            train_dataset = dataset_manager.load_training_dataset(dataset_name, dataset_limit)
            dataset_manager.display_dataset_preview(train_dataset)
            
            # Train model
            trainer = sft_manager.train_model(model, tokenizer, train_dataset)
            
            # Move to CPU if not using GPU
            if not use_gpu:
                trainer.model.to("cpu")
            
            # Test trained model
            ModelTester.test_model_with_questions(
                trainer.model, tokenizer, test_questions, title="Small Model (After SFT) Output"
            )
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
